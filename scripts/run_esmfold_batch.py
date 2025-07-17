import os
import torch
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from esm import pretrained

# === Settings ===
FASTA_PATH = "data/decoded_structdiff.fasta"
OUT_DIR = "data/esmfold_outputs"
BATCH_SIZE = 5  # ✅ adjust this number
os.makedirs(OUT_DIR, exist_ok=True)

# === Load model ===
model = pretrained.esmfold_v1()
model = model.eval().cuda() if torch.cuda.is_available() else model.eval()

# === Load sequences ===
records = list(SeqIO.parse(FASTA_PATH, "fasta"))

# === Process in batches ===
total_results = []
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i+BATCH_SIZE]
    batch_results = []

    for record in tqdm(batch, desc=f"🔬 Processing batch {i//BATCH_SIZE + 1}"):
        name = record.id
        sequence = str(record.seq).replace(" ", "").replace("\n", "")
        if len(sequence) < 20 or "X" in sequence:
            continue

        try:
            with torch.no_grad():
                structure = model.infer_pdb(sequence)
                output_path = os.path.join(OUT_DIR, f"{name}.pdb")
                with open(output_path, "w") as f:
                    f.write(structure)

                output = model.predict(sequence)
                plddt = output["plddt"].detach().cpu().numpy().mean()

            batch_results.append({
                "Sequence_ID": name,
                "Sequence": sequence,
                "pLDDT": round(plddt, 2),
                "PDB_Path": output_path
            })
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")

    # ✅ Save after each batch
    df = pd.DataFrame(batch_results)
    df.to_csv(os.path.join(OUT_DIR, f"batch_{i//BATCH_SIZE + 1}_scores.csv"), index=False)
    total_results.extend(batch_results)

# === Save full results ===
pd.DataFrame(total_results).to_csv(os.path.join(OUT_DIR, "esmfold_batch_scores.csv"), index=False)
print("✅ All done!")
