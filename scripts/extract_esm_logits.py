# scripts/extract_esm_logits.py

import os
import sys
import torch
from tqdm import tqdm
import esm  # from fair-esm

# === Fix PYTHONPATH and paths ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.makedirs("data", exist_ok=True)

# === Load decoder dataset ===
data_path = "data/decoder_dataset.pt"
data = torch.load(data_path)
sequences = data["sequences"]  # Tensor (B, L)
vocab = {
    0: "-",  # <PAD>
    1: "A", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "K", 10: "L", 11: "M", 12: "N", 13: "P", 14: "Q", 15: "R",
    16: "S", 17: "T", 18: "V", 19: "W", 20: "Y", 21: "*", 22: "<START>"
}

# === Decode token sequences to strings ===
def decode_tokens(toks):
    return "".join(vocab[i] for i in toks if i not in (0, 21, 22))  # remove PAD, EOS, START

decoded_seqs = [decode_tokens(seq.tolist()) for seq in sequences]

# === Load ESM2 model from fair-esm ===
print("🔄 Loading ESM2 model from fair-esm...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# === Map from ESM alphabet to our decoder vocab (size 23) ===
token_to_idx = alphabet.tok_to_idx
decoder_tokens = [
    "<pad>", "A", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "<eos>", "<cls>"
]
esm2_to_decoder_idx = [token_to_idx[tok] for tok in decoder_tokens]


# === Extract logits ===
print(f"🔬 Extracting ESM2 logits for {len(decoded_seqs)} sequences...")
esm_logits = []

with torch.no_grad():
    for seq in tqdm(decoded_seqs, desc="ESM2", ncols=80):
        batch = [("sequence", seq)]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)

        output = model(tokens, repr_layers=[], return_contacts=False)
        per_tok_logits = output["logits"][0, 1:-1].cpu()  # (L, V)

        # Project ESM2 logits to decoder vocab space
        projected_logits = per_tok_logits[:, esm2_to_decoder_idx]  # (L, 23)
        esm_logits.append(projected_logits)

# === Pad logits to max sequence length ===
max_len = sequences.size(1)
vocab_size = 23
padded_logits = torch.zeros((len(esm_logits), max_len, vocab_size))

for i, logits in enumerate(esm_logits):
    L = logits.size(0)
    padded_logits[i, :L, :] = logits

# === Save logits for distillation ===
torch.save(padded_logits, "data/esm_logits.pt")
print("✅ Saved teacher logits: data/esm_logits.pt")
