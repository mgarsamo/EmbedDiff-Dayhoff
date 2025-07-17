import os
import pandas as pd

# === File paths ===
CURATED_FASTA_RAW = "/Users/melaku/Documents/Projects/StructDiff/data/curated_sequences.fasta"
STRUCTDIFF_FASTA = "/Users/melaku/Documents/Projects/StructDiff/data/decoded_structdiff.fasta"
CURATED_CSV = "/Users/melaku/Documents/Projects/StructDiff/data/curated_sequences.csv"
CURATED_FASTA_CLEAN = CURATED_FASTA_RAW.replace(".fasta", "_cleaned.fasta")
BLAST_DB = "/Users/melaku/Documents/Projects/StructDiff/data/real_db"
BLAST_OUTPUT = "/Users/melaku/Documents/Projects/StructDiff/data/blast_structdiff_vs_real.tsv"
OUTPUT_CSV = "/Users/melaku/Documents/Projects/StructDiff/figures/blast_structdiff_matches.csv"

# === Step 0: Clean FASTA headers ===
print("🧼 Cleaning curated FASTA headers...")
with open(CURATED_FASTA_RAW, "r", encoding="utf-8") as infile, \
     open(CURATED_FASTA_CLEAN, "w", encoding="ascii", errors="ignore") as outfile:
    for line in infile:
        if line.startswith(">"):
            header = line.strip().replace(">", "").split()[0]
            outfile.write(f">{header}\n")
        else:
            outfile.write(line)

# === Step 1: Build BLAST DB ===
if not os.path.exists(BLAST_DB + ".pin"):
    print("🔨 Building BLAST database from curated sequences...")
    os.system(f"makeblastdb -in {CURATED_FASTA_CLEAN} -dbtype prot -out {BLAST_DB}")
else:
    print("📂 BLAST database already exists. Skipping rebuild.")

# === Step 2: Run BLAST ===
print("🚀 Running BLAST...")
blast_cmd = (
    f"blastp -query {STRUCTDIFF_FASTA} -db {BLAST_DB} "
    f"-outfmt '6 qseqid sseqid pident evalue' "
    f"-max_target_seqs 1 -evalue 1e-5 "
    f"-out {BLAST_OUTPUT}"
)
os.system(blast_cmd)

# === Step 3: Load BLAST results ===
print("📊 Parsing BLAST output...")
blast_df = pd.read_csv(BLAST_OUTPUT, sep="\t", header=None)
blast_df.columns = ["Generated_ID", "Matched_Real_Uniprot", "Percent_Identity", "E_value"]

# === Step 4: Load curated metadata ===
meta_df = pd.read_csv(CURATED_CSV, encoding="ISO-8859-1")

# === Step 5: Extract StructDiff sequences
gen_sequences = {}
with open(STRUCTDIFF_FASTA, "r") as f:
    current_id = ""
    seq_lines = []
    for line in f:
        if line.startswith(">"):
            if current_id:
                gen_sequences[current_id] = "".join(seq_lines)
            current_id = line.strip().replace(">", "")
            seq_lines = []
        else:
            seq_lines.append(line.strip())
    if current_id:
        gen_sequences[current_id] = "".join(seq_lines)

blast_df["Generated_Sequence"] = blast_df["Generated_ID"].map(gen_sequences)

# === Step 6: Join metadata for matched real sequences
blast_df = blast_df.merge(meta_df, left_on="Matched_Real_Uniprot", right_on="uniprot_id", how="left")
blast_df.rename(columns={
    "name": "Matched_Real_Name",
    "class": "Matched_Real_Class",
    "chain_seq": "Matched_Real_Sequence"
}, inplace=True)

# === Step 7: Final columns
final_df = blast_df[[
    "Generated_ID", "Generated_Sequence",
    "Matched_Real_Uniprot", "Matched_Real_Name", "Matched_Real_Class", "Matched_Real_Sequence",
    "Percent_Identity", "E_value"
]]

# === Step 8: Save final output
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Final BLAST matches saved to: {OUTPUT_CSV}")
print(final_df.head())
