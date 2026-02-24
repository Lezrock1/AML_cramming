import math
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from safetensors.torch import load_file


def _ensure_config_json(path):
    """Cramming saves 'model_config.json', but HF Transformers expects 'config.json'.
    Create a symlink automatically if needed."""
    config_json = os.path.join(path, "config.json")
    model_config_json = os.path.join(path, "model_config.json")
    if not os.path.exists(config_json) and os.path.exists(model_config_json):
        os.symlink("model_config.json", config_json)
        print(f"[INFO] Created symlink {config_json} -> model_config.json")


@torch.no_grad()
def wikitext2_perplexity(model_name_or_path, device="cuda", split="test", max_length=128, stride=128):
    _ensure_config_json(model_name_or_path)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.model_max_length = 10**9

    config = LlamaConfig.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM(config)

    state_dict = load_file(os.path.join(model_name_or_path, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing", len(missing))
    print("unexpected", len(unexpected))
    print("missing[:20]", missing[:20])
    print("unexpected[:20]", unexpected[:20])

    if missing or unexpected:
        print(f"[WARN] missing={len(missing)} unexpected={len(unexpected)}")
        # ggf. einmal ausgeben zum Debuggen:
        # print("missing:", missing[:20]); print("unexpected:", unexpected[:20])

    model = model.to(device).eval()

    w = model.model.embed_tokens.weight
    print("embed mean abs:", w.abs().mean().item(), "std:", w.std().item())


    # --- SANITY CHECK: toy perplexity ---
    s = "Hello world!"
    print("tokens:", tok.tokenize(s))
    print("ids:", tok(s, add_special_tokens=False)["input_ids"][:30])
    print("unk_id:", tok.unk_token_id, "pad_id:", tok.pad_token_id)

    texts = ["Hello world!", "The quick brown fox jumps over the lazy dog."]
    enc_toy = tok("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)
    enc_toy.pop("token_type_ids", None)   # <- FIX
    enc_toy = {k: v.to(device) for k, v in enc_toy.items()}
    out_toy = model(**enc_toy, labels=enc_toy["input_ids"])
    print("toy loss:", out_toy.loss.item(), "toy ppl:", math.exp(out_toy.loss.item()))
    # --- end sanity check ---

    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0

    # overlap = max_length - stride (bei stride=max_length ist overlap=0)
    for start in range(0, seq_len, stride):
        end = min(start + max_length, seq_len)
        if end - start < 2:
            break

        input_chunk = input_ids[:, start:end]
        labels = input_chunk.clone()

        # Always skip the first token (no left context in window)
        labels[:, 0] = -100

        # When using overlap (stride < max_length): only evaluate the “new” tokens
        if stride < max_length and start > 0:
            overlap = max_length - stride
            # Mask overlapping context tokens (from token 1 to overlap)
            labels[:, 1:overlap+1] = -100

        out = model(input_ids=input_chunk, labels=labels)

        valid = (labels != -100).sum().item()
        nll_sum += out.loss.item() * valid
        n_tokens += valid

        if end == seq_len:
            break

    ppl = math.exp(nll_sum / max(1, n_tokens))
    return float(ppl)

if __name__ == "__main__":
    path = "outputs/TESTS_DICT/crambert_LLaMA_12h_v3/checkpoints/LlamaFromAML_2026-02-04_0.0385"
    print(f"Testing: {path}")
    ppl = wikitext2_perplexity(
        path,
        max_length=128,
        stride=64,
    )
    print("WikiText-2 PPL:", ppl)

    path = "outputs/LLaMAcram_12h_wcl/checkpoints/LlamaFromAML_2026-02-05_0.0282"
    print(f"Testing: {path}")
    ppl = wikitext2_perplexity(
        path,
        max_length=128,
        stride=64,
    )
    print("WikiText-2 PPL:", ppl)

    path = "outputs/LLaMAcram_24h_wcl/checkpoints/LlamaFromAML_2026-02-06_0.0273"
    print(f"Testing: {path}")
    ppl = wikitext2_perplexity(
        path,
        max_length=128,
        stride=64,
    )
    print("WikiText-2 PPL:", ppl)

    path = "outputs/LLaMAcram_24h_wcl_LR001/checkpoints/LlamaFromAML_2026-02-09_0.0284"
    print(f"Testing: {path}")
    ppl = wikitext2_perplexity(
        path,
        max_length=128,
        stride=64,
    )
    print("WikiText-2 PPL:", ppl)

    path = "outputs/LLaMAcram_8h_H100_wcl_256/checkpoints/LlamaFromAML_2026-02-09_0.1124"
    print(f"Testing: {path}")
    ppl = wikitext2_perplexity(
        path,
        max_length=256,
        stride=128,
    )
    print("WikiText-2 PPL:", ppl)

    path = "outputs/LLaMAbase_24h/checkpoints/LlamaFromAML_2026-02-11_0.0318"
    print(f"Testing: {path}")
    ppl = wikitext2_perplexity(
        path,
        max_length=128,
        stride=64,
    )
    print("WikiText-2 PPL:", ppl)
