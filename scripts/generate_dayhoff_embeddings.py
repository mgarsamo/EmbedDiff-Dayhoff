# scripts/generate_dayhoff_embeddings.py

import os
import torch
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Paths ===
FASTA_PATH = "data/decoded_embeddiff.fasta"
OUTPUT_PATH = "embeddings/decoded_dayhoff_embeddings.npy"

# === Load Dayhoff model ===
MODEL_NAME = "microsoft/Dayhoff-3b-UR90"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()

# === Load sequences ===
records = list(SeqIO.parse(FASTA_PATH, "fasta"))
data = [str(record.seq) for record in records]

# === Generate embeddings ===
sequence_representations = []
with torch.no_grad():
    for i in tqdm(range(0, len(data), 8), desc="🔬 Embedding sequences with Dayhoff"):
        batch = data[i:i+8]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, 1280)

        # mean pool over tokens, excluding BOS/EOS
        mask = attention_mask.clone()
        if mask.size(1) > 0:
            mask[:, 0] = 0  # drop BOS
        lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)
        eos_idx = lengths - 1
        rows = torch.arange(input_ids.size(0))
        mask[rows, eos_idx] = 0  # drop EOS

        mask = mask.to(hidden.dtype)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        sequence_representations.extend(pooled.cpu().numpy())

# === Save to disk ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
arr = np.array(sequence_representations, dtype=np.float32)

if arr.shape[1] != 1280:
    raise RuntimeError(f"Expected 1280-d embeddings, got {arr.shape[1]}")

np.save(OUTPUT_PATH, arr)
print(f"✅ Saved Dayhoff embeddings {arr.shape} to {OUTPUT_PATH}")
