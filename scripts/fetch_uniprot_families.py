import os
import requests
from Bio import SeqIO
from io import StringIO
from collections import OrderedDict

# === Configuration ===
PROTEIN_NAME = "thioredoxin reductase"
ORGANISMS = ["Bacteria", "Fungi", "Archaea"]
MIN_LEN = 200
MAX_LEN = 350
LIMIT = 500         # UniProt query max
DESIRED_N = 100     # Target per organism

# === Output paths ===
OUT_DIR = "data"
FINAL_FASTA = os.path.join(OUT_DIR, "curated_thioredoxin_reductase.fasta")
os.makedirs(OUT_DIR, exist_ok=True)

# === Initialize storage
all_records = OrderedDict()
seen_ids = set()

def fetch_sequences(protein: str, organism: str):
    """
    Fetch protein sequences from UniProt for a specific organism and protein name.

    Args:
        protein (str): Protein name to search
        organism (str): Taxonomic group

    Returns:
        list: Unique SeqRecord objects with class tag added
    """
    print(f"\n🔍 Fetching {protein} sequences from {organism}...")

    query = f'(protein_name:"{protein}") AND (taxonomy_name:"{organism}") AND (length:[{MIN_LEN} TO {MAX_LEN}])'
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "fasta",
        "size": LIMIT
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"❌ Failed to fetch from UniProt for {organism} — Status: {response.status_code}")
        return []

    records = list(SeqIO.parse(StringIO(response.text), "fasta"))

    unique = OrderedDict()
    for record in records:
        uid = record.id.split("|")[1] if "|" in record.id else record.id
        seq = str(record.seq)
        if uid not in seen_ids and seq not in unique:
            seen_ids.add(uid)
            record.description += f" [{organism.lower()}]"  # Add class label
            unique[seq] = record
        if len(unique) >= DESIRED_N:
            break

    return list(unique.values())

# === Fetch and collect
for org in ORGANISMS:
    records = fetch_sequences(PROTEIN_NAME, org)
    for record in records:
        seq = str(record.seq)
        if seq not in all_records:
            all_records[seq] = record

# === Save output
SeqIO.write(list(all_records.values()), FINAL_FASTA, "fasta")
print(f"\n✅ Final curated FASTA saved: {FINAL_FASTA} ({len(all_records)} sequences total)")
