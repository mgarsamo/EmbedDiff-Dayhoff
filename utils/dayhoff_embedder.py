
# utils/dayhoff_embedder.py

import os
import numpy as np
import torch
from tqdm import tqdm
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def _mean_pool_last_hidden(hidden, input_ids, attention_mask, tokenizer, exclude_special=True):
    """
    Mean-pool hidden states over real residue tokens (ignoring BOS/EOS/pad).
    Returns (B, D) float tensor.
    """
    # hidden is already the last_hidden_state tensor (B, L, D)
    mask = attention_mask.clone()       # (B, L)

    if exclude_special:
        # Drop BOS (first token)
        if mask.size(1) > 0:
            mask[:, 0] = 0
        # Drop EOS (last non-pad token per row)
        if tokenizer.pad_token_id is not None:
            lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)
        else:
            lengths = torch.full((input_ids.size(0),), input_ids.size(1), device=input_ids.device)
        eos_idx = lengths - 1
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        mask[rows, eos_idx] = 0

    mask = mask.to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled

@torch.no_grad()
def embed_sequences(fasta_file, output_npy="embeddings/dayhoff_embeddings.npy",
                    batch_size=8, device="cpu"):
    """
    Embed protein sequences from FASTA using Dayhoff (microsoft/Dayhoff-3b-UR90).
    Save as (N, 1280) float32 NumPy array.
    """
    sequences = [str(r.seq) for r in SeqIO.parse(fasta_file, "fasta")]
    assert sequences, "No sequences found in FASTA."

    model_name = "microsoft/Dayhoff-3b-UR90"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_mamba_kernels=False)
    model.eval().to(device)

    all_chunks = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="🔬 Embedding with Dayhoff"):
        batch = sequences[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # For Jamba models, use the last hidden state from hidden_states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1]
        else:
            last_hidden = outputs.last_hidden_state
        pooled = _mean_pool_last_hidden(last_hidden, input_ids, attention_mask, tokenizer)
        all_chunks.append(pooled.cpu().numpy())

    arr = np.concatenate(all_chunks, axis=0).astype(np.float32)

    # sanity check: Dayhoff 3B model hidden_size should be 1280
    if arr.shape[1] != 1280:
        raise RuntimeError(f"Expected 1280-d embeddings, got {arr.shape[1]}")

    # make sure output folder exists
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)

    np.save(output_npy, arr)
    print(f"✅ Saved Dayhoff embeddings {arr.shape} to {output_npy}")
    return arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed protein sequences using Dayhoff model")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output NPY file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    print(f"🔬 Loading sequences from {args.input}")
    embed_sequences(args.input, args.output, args.batch_size, args.device)
