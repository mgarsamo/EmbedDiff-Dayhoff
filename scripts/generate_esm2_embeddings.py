# scripts/generate_esm2_embeddings.py

import torch
import esm
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import os

# === Paths ===
FASTA_PATH = "data/decoded_embeddiff.fasta"
OUTPUT_PATH = "embeddings/decoded_embeddings.npy"

# === Load ESM-2 model ===
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# === Load sequences ===
records = list(SeqIO.parse(FASTA_PATH, "fasta"))
data = [(record.id, str(record.seq)) for record in records]

# === Convert sequences to tokens ===
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# === Generate embeddings ===
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
token_representations = results["representations"][33]

# === Average embeddings per sequence ===
sequence_representations = []
for i, (_, seq) in tqdm(enumerate(data), total=len(data), desc="🔬 Embedding sequences"):
    start, end = 1, len(seq) + 1
    emb = token_representations[i, start:end].mean(0).numpy()
    sequence_representations.append(emb)

# === Save to disk ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.save(OUTPUT_PATH, np.array(sequence_representations))
print(f"✅ Saved ESM2 embeddings to {OUTPUT_PATH}")
