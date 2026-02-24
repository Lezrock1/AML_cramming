import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

hf_dataset_id = "JonasGeiping/the_pile_WordPiecex32768_2efdb9d060d1ae95faf952ec1a50f020"

# lädt nur den tokenizer/-Ordner aus dem Dataset-Repo in den HF-Cache
local_ds_dir = snapshot_download(
    repo_id=hf_dataset_id,
    repo_type="dataset",
    allow_patterns=["tokenizer/*"],
)

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(local_ds_dir, "tokenizer"),
    use_fast=True,
)

print("loaded vocab_size:", tokenizer.vocab_size)
