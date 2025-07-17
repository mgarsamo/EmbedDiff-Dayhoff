import os
import shutil
import torch
from Bio import SeqIO
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from esm import pretrained

import esm
from esm import pretrained

print("✅ ESM is now correctly installed!")
model = pretrained.esmfold_v1()
print("✅ ESMFold model loaded successfully.")


# === Config & Paths ===
FASTA_PATH = "/Users/melaku/Documents/Projects/StructDiff/data/decoded_structdiff_filtered.fasta"
OUT_DIR = "esmfold_output"
MAX_LEN = 1024
N_SEQUENCES = 20

# === Prepare output directory ===
if os.path.exists(OUT_DIR):
    print(f"🧹 Removing old output folder `{OUT_DIR}`")
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# === Load pretrained ESMFold model (automatically handles config + checkpoint) ===
print("🧠 Loading ESMFold model on CPU...")
model = pretrained.esmfold_v1()
model = model.eval().cpu()

# === Helpers ===
def sanitize(name):
    return re.sub(r'[^A-Za-z0-9_\\-]', '_', name)

def rate_plddt(x):
    if x >= 90: return "High"
    elif x >= 70: return "Medium"
    else: return "Low"

# === Load and subset sequences ===
all_recs = list(SeqIO.parse(FASTA_PATH, "fasta"))
records = all_recs[:N_SEQUENCES]
print(f"📥 Loaded {len(records)} / {len(all_recs)} sequences (limiting to {N_SEQUENCES})")

# === Run folding ===
results = []
print("🔄 Running ESMFold inference...\n")
for rec in tqdm(records, desc="🧬 Folding", unit="seq", ncols=80):
    seq_id = sanitize(rec.id)
    seq = str(rec.seq)

    if not set(seq).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
        print(f"⚠️ Skipping {seq_id}: invalid characters")
        continue
    if len(seq) > MAX_LEN:
        print(f"⚠️ Skipping {seq_id}: length > {MAX_LEN}")
        continue

    try:
        with torch.no_grad():
            out = model.infer(seq)
            pdb = out["pdb"]
            plddt = out["plddt"]
            mean = float(np.mean(plddt))
            rating = rate_plddt(mean)
    except Exception as e:
        print(f"❌ Fold failed for {seq_id}: {e}")
        continue

    # Save PDB
    pdb_file = os.path.join(OUT_DIR, f"{seq_id}.pdb")
    with open(pdb_file, "w") as f:
        f.write(pdb)

    # Record metrics
    results.append({
        "Sequence_ID": seq_id,
        "Sequence_Length": len(seq),
        "Mean_pLDDT": mean,
        "pLDDT_Rating": rating
    })

# === Write CSV ===
df = pd.DataFrame(results)
csv_out = os.path.join(OUT_DIR, "esmfold_metrics.csv")
df.to_csv(csv_out, index=False)

print(f"\n✅ Done! Structures in `{OUT_DIR}/` and metrics in `{csv_out}`")
