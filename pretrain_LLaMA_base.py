import os, sys, time, math, random, json, datetime
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

class _Tee:
    """Duplicates writes to multiple streams (console + log file)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, obj):
        for s in self.streams:
            s.write(obj)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def make_cosine_warmup_scheduler(optimizer, total_steps, warmup_steps=2000, min_lr_ratio=0.1):
    """
    Standard LLaMA scheduler: linear warmup then cosine decay to min_lr_ratio * peak_lr.
    (Touvron et al., 2023)
    """
    def lr_factor(step_idx: int):
        step = step_idx + 1
        if step <= warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

def causal_lm_collate_tokenized(examples, tokenizer, seq_length=128):
    """
    examples: list of dicts from streaming dataset
      each example should contain "input_ids": list[int] (len=seq_length)
    """
    input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)

    if input_ids.size(1) != seq_length:
        input_ids = input_ids[:, :seq_length]
        if input_ids.size(1) < seq_length:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            pad = torch.full((input_ids.size(0), seq_length - input_ids.size(1)), pad_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad], dim=1)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = (input_ids != pad_id).long()

    # Causal LM labels: predict next token (model shifts internally)
    labels = input_ids.clone()
    labels[input_ids == pad_id] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def save_ckpt(path, model, optim, scaler, step, elapsed):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "meta": {"step": step, "elapsed": elapsed},
    }, path)

def load_ckpt(path, model, optim, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    # Optimizer/scaler state can become incompatible if parameter groups change.
    try:
        optim.load_state_dict(ckpt["optim"])
    except Exception as e:
        print(f"[WARN] Could not load optimizer state ({type(e).__name__}: {e}). Continuing with fresh optimizer.")
    if scaler and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[WARN] Could not load scaler state ({type(e).__name__}: {e}). Continuing with fresh scaler.")
    meta = ckpt.get("meta", {})
    return int(meta.get("step", 0)), float(meta.get("elapsed", 0.0))


def _make_adamw_param_groups(model):
    """Create AdamW parameter groups with weight decay disabled for norms and biases.

    LLaMA-style recipes typically apply weight decay to weight matrices but not to
    biases and normalization parameters.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_l = name.lower()
        is_bias = name_l.endswith(".bias")
        is_norm = ("norm" in name_l) or ("layernorm" in name_l)
        # 1D params are almost always bias / norm scale.
        if is_bias or is_norm or param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def train(
    hf_dataset_id="JonasGeiping/the_pile_WordPiecex32768_2efdb9d060d1ae95faf952ec1a50f020",
    base_dir="outputs",
    run_name="LLaMAbase_24h",
    micro_batch_size=64,
    print_every=1000,
    budget_hours=24,
    max_len=128,
    use_amp=True,
    gradient_clipping=None):       # LLaMA-style: no clipping

    # on A30:
    # max_len=128 crashed with: 256, 128 (VRAM exceeded) -> use micro_batch_size=64

    # on H100:
    # max_len=128 crashed with: 2048, 1024, 512 (VRAM exceeded) -> use micro_batch_size=256
    # max_len=256 crashed with: 1024, 512, 256 (VRAM exceeded) -> use micro_batch_size=128
    
    batch_size = 8192
    peak_lr = 3e-4                 # LLaMA default for small models
    warmup_steps = 2000            # LLaMA default

    if batch_size is None:
        grad_accum_steps = 1
    else:
        grad_accum_steps = max(1, int(batch_size // micro_batch_size))

    run_dir = os.path.join(base_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "intermediate_state.pth")

    # ---- redirect stdout to console + log file (same layout as pretrain.py) ----
    now = datetime.datetime.now()
    pretrain_log_dir = os.path.join(run_dir, "pretrain", now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S"))
    os.makedirs(pretrain_log_dir, exist_ok=True)
    log_path = os.path.join(pretrain_log_dir, f"{run_name}_pretrain.log")
    _log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    print(f"\n===== Training log started at {now.strftime('%Y-%m-%d %H:%M:%S')} =====")
    print(f"Log file: {log_path}")

    print(
        "Train config:",
        {
            "hf_dataset_id": hf_dataset_id,
            "base_dir": base_dir,
            "run_name": run_name,
            "micro_batch_size": micro_batch_size,
            "print_every": print_every,
            "budget_hours": budget_hours,
            "max_len": max_len,
            "use_amp": use_amp,
            "gradient_clipping": gradient_clipping,
            "batch_size": batch_size,
            "peak_lr": peak_lr,
            "warmup_steps": warmup_steps,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    local_ds_dir = snapshot_download(
        repo_id=hf_dataset_id,
        repo_type="dataset",
        allow_patterns=["tokenizer/*"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(local_ds_dir, "tokenizer"),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    print("loaded vocab_size:", tokenizer.vocab_size)

    cfg = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=16,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(cfg).to(device)
    print('Total parameters: ', sum(p.numel() for p in model.parameters()))

    # Standard LLaMA: AdamW with beta1=0.9, beta2=0.95, weight_decay=0.1
    # Use peak_lr as base so LambdaLR scales correctly.
    param_groups = _make_adamw_param_groups(model)
    optim = torch.optim.AdamW(param_groups, lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.0)
    decay_elems = sum(p.numel() for p in param_groups[0]["params"])
    no_decay_elems = sum(p.numel() for p in param_groups[1]["params"])
    print(f"AdamW param groups: decay={decay_elems:,} params, no_decay={no_decay_elems:,} params")
    scheduler = None  # created after measuring actual step speed
    _optim_step_times = []  # track wall-time per optimizer step
    _optim_step_t0 = None

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    ds = load_dataset(hf_dataset_id, split="train", streaming=True, token=os.environ.get("HF_TOKEN"))
    ex = next(iter(ds))
    print(ex.keys())
    print(type(ex["input_ids"]), len(ex["input_ids"]), ex["input_ids"][:10])

    collate = lambda b: causal_lm_collate_tokenized(b, tokenizer, seq_length=max_len)

    dl = DataLoader(
        ds,
        batch_size=micro_batch_size,
        #shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )

    it = iter(dl)

    step0, elapsed = 0, 0.0
    if os.path.isfile(ckpt_path):
        try:
            step0, elapsed = load_ckpt(ckpt_path, model, optim, scaler)
            print(f"Resumed: step={step0}, elapsed={elapsed:.1f}s")
        except Exception:
            print("Checkpoint corrupt -> start fresh")
            os.remove(ckpt_path)

    wallclock = time.time() - elapsed
    t_print = time.time()
    loss_buf = []

    model.train()
    optim.zero_grad(set_to_none=True)
    step = step0
    while True:
        step += 1
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
            out = model(**batch)
            loss = out.loss / grad_accum_steps

        if not torch.isfinite(loss):
            print(f"Non-finite loss at step {step}. Abort or reload.")
            break

        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            # Gradient Clipping
            if gradient_clipping is not None and gradient_clipping > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            
            scaler.step(optim)
            scaler.update()

            # Track optimizer step durations
            if _optim_step_t0 is not None:
                _optim_step_times.append(time.time() - _optim_step_t0)
            _optim_step_t0 = time.time()

            # Create scheduler using avg of optimizer steps 4-6 (skip CUDA warmup)
            if scheduler is None and len(_optim_step_times) >= 6:
                secs_per_optim_step = sum(_optim_step_times[3:6]) / 3.0
                optim_steps = max(1, int(budget_hours * 3600 / secs_per_optim_step))
                scheduler = make_cosine_warmup_scheduler(optim, total_steps=optim_steps, warmup_steps=min(warmup_steps, optim_steps // 2))
                print(f"Measured {secs_per_optim_step:.2f}s/optim_step (avg of steps 4-6) -> ~{optim_steps} optimizer steps for {budget_hours}h budget")

            if scheduler is not None:
                scheduler.step()
            optim.zero_grad(set_to_none=True)

        loss_buf.append(loss.detach().float().cpu())

        if step % print_every == 0:
            avg = torch.stack(loss_buf).mean().item()
            dt = (time.time() - t_print) / print_every
            toks = batch["input_ids"].numel()
            current_lr = scheduler.get_last_lr()[0] if scheduler else peak_lr
            train_time = (time.time() - wallclock)/60/60
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_loss = loss.item() * grad_accum_steps
            print(f"{current_time} | elapsed {train_time:.2f}std | step {step} | loss {current_loss:.4f} | avg {avg:.4f} | lr {current_lr:.6f} | {dt:.3f}s/step | ~{toks/dt:.0f} tok/s")
            loss_buf.clear()
            t_print = time.time()
        
            # Save training state for resume (intermediate checkpoint)
            save_ckpt(ckpt_path, model, optim, scaler, step, time.time() - wallclock)

        if (time.time() - wallclock) / 3600.0 > budget_hours:
            print("Budget reached -> save & stop")
            save_ckpt(ckpt_path, model, optim, scaler, step, time.time() - wallclock)
            break

    # Save final model (like pretrain.py save_final_model)
    from safetensors.torch import save_file
    
    if torch.isfinite(loss):
        now = datetime.datetime.now()
        long_checkpoint_id = f"LlamaFromAML_{now.strftime('%Y-%m-%d')}_{loss.item():2.4f}"
        full_path = os.path.join(base_dir, run_name, "checkpoints", long_checkpoint_id)
        os.makedirs(full_path, exist_ok=True)
        
        # Save like pretrain.py: tokenizer, model.safetensors, model_config.json
        tokenizer.save_pretrained(full_path)
        # Clone & contiguous to handle weight tying
        state_dict = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, os.path.join(full_path, "model.safetensors"))
        with open(os.path.join(full_path, "model_config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    
    # Cleanup to avoid PyGIL errors at exit
    del model, optim, scheduler, scaler, batch, dl, it, loss_buf
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Job complete and cleaned up.")

    # ---- restore stdout & close log file ----
    sys.stdout = sys.__stdout__
    _log_file.close()

if __name__ == "__main__":
    train()
