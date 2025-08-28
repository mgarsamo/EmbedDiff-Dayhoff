# scripts/build_decoder_dataset_dayhoff.py

import os
import torch
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# === SETTINGS ===
FASTA_PATH = "data/curated_thioredoxin_reductase.fasta"
OUTPUT_PT = "data/decoder_dataset_dayhoff.pt"
MAX_SEQ_LEN = 350
BATCH_SIZE = 4
NOISE_STD = 0.03  # ✅ Gentler noise, more realistic variation

# === Load sequences ===
records = list(SeqIO.parse(FASTA_PATH, "fasta"))
sequences = [str(r.seq) for r in records]
ids = [r.id for r in records]
descriptions = [r.description for r in records]
print(f"📄 Loaded {len(sequences)} sequences.")

# === Infer classes from description ===
def infer_class(description):
    desc = description.lower()
    if "lactamase" in desc:
        return "beta_lactamase"
    elif "protease" in desc or "trypsin" in desc or "elastase" in desc:
        return "serine_protease"
    elif "kinase" in desc:
        return "kinase"
    else:
        return "unknown"

class_labels = [infer_class(d) for d in descriptions]
classes = sorted(set(c for c in class_labels if c != "unknown"))
class_to_idx = {cls: i for i, cls in enumerate(classes)}

class_tensor = torch.zeros(len(class_labels), len(classes)).float()
for i, label in enumerate(class_labels):
    if label in class_to_idx:
        class_tensor[i, class_to_idx[label]] = 1.0

# === Load Dayhoff model ===
MODEL_NAME = "microsoft/Dayhoff-3b-UR90"
print(f"📦 Loading Dayhoff model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, use_mamba_kernels=False)
model.eval()

# === Amino acid token mapping (manual fallback tokenizer) ===
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_id = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
aa_to_id["<PAD>"] = 0
aa_to_id["<EOS>"] = 21

def tokenize(seq, max_len=MAX_SEQ_LEN):
    tokens = [aa_to_id.get(aa, 0) for aa in seq[:max_len - 1]]
    tokens.append(aa_to_id["<EOS>"])
    tokens += [aa_to_id["<PAD>"]] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long)

# === Collect outputs ===
all_token_embeddings = []
all_logits = []
all_tokens = []

print("🔬 Generating token-level embeddings and logits (Dayhoff)...")
with torch.no_grad():
    for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_seqs = sequences[i:i + BATCH_SIZE]

        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # === (B, L, 1280) token-level embeddings from last_hidden_state
        # For Jamba models, use the last hidden state from hidden_states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1]
        else:
            last_hidden = outputs.last_hidden_state
        token_emb = last_hidden[:, :MAX_SEQ_LEN, :]
        token_emb += torch.randn_like(token_emb) * NOISE_STD  # Add mild noise

        # === Pad embeddings if needed
        pad_len_emb = MAX_SEQ_LEN - token_emb.size(1)
        if pad_len_emb > 0:
            pad_emb_shape = (token_emb.size(0), pad_len_emb, token_emb.size(2))
            token_emb = torch.cat([token_emb, torch.zeros(pad_emb_shape)], dim=1)
        all_token_embeddings.append(token_emb)

        # === (B, L, V) token logits from model output
        logits = outputs.logits[:, :MAX_SEQ_LEN, :]
        pad_len = MAX_SEQ_LEN - logits.size(1)
        if pad_len > 0:
            pad_shape = (logits.size(0), pad_len, logits.size(2))
            logits = torch.cat([logits, torch.zeros(pad_shape)], dim=1)
        all_logits.append(logits)

        # === Manual tokenization for sequence tensor
        toks = [tokenize(seq) for seq in batch_seqs]
        all_tokens.extend(toks)

# === Stack tensors
token_embeddings_tensor = torch.cat(all_token_embeddings, dim=0)   # (N, L, 1280)
dayhoff_logits_tensor = torch.cat(all_logits, dim=0)               # (N, L, V)
sequence_tensor = torch.stack(all_tokens)                          # (N, L)

# === Save to disk
os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
torch.save({
    "token_embeddings": token_embeddings_tensor,
    "sequences": sequence_tensor,
    "dayhoff_logits": dayhoff_logits_tensor,
    "class_onehot": class_tensor,
    "descriptions": descriptions,  # ✅ Optional: traceability
    "ids": ids
}, OUTPUT_PT)

print(f"✅ Dayhoff token-level decoder dataset saved to: {OUTPUT_PT}")
