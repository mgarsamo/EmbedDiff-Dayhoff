# Activate your .venv
#source .venv/bin/activate

import os
import subprocess
import argparse
import sys

def run_command(command, description, skip_steps, step_key):
    if step_key in skip_steps:
        print(f"⏭️ Skipping: {description}")
        return
    print(f"\n✅ {description}")
    print(f"🔧 Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Failed at: {description}")

def main():
    parser = argparse.ArgumentParser(description="Run the full EmbedDiff-Dayhoff pipeline.")
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="List of step keys to skip (e.g. dayhoff tsne diffusion decoder blast html)"
    )
    args = parser.parse_args()
    skip_steps = set(args.skip)

    fasta_path = "data/curated_thioredoxin_reductase.fasta"
    print(f"\n✅ Dataset prepared at: {fasta_path}")

    run_command(
        f"python utils/dayhoff_embedder.py --input {fasta_path} --output embeddings/dayhoff_embeddings.npy",
        "Step 2a: Dayhoff embedding of real sequences",
        skip_steps, "dayhoff"
    )

    run_command(
        "python scripts/plot_tsne_by_domain_dayhoff.py",
        "Step 2b: Plot t-SNE of real Dayhoff embeddings",
        skip_steps, "tsne"
    )

    run_command(
        "python scripts/train_embeddiff_dayhoff.py",
        "Step 3: Train EmbedDiff-Dayhoff latent diffusion model",
        skip_steps, "diffusion"
    )

    run_command(
        "python scripts/sample_embeddings_dayhoff.py",
        "Step 4: Sample synthetic embeddings from EmbedDiff-Dayhoff",
        skip_steps, "sample"
    )

    run_command(
        "python scripts/build_decoder_dataset_dayhoff.py",
        "Step 5a: Build decoder dataset from real Dayhoff embeddings",
        skip_steps, "decoder_data"
    )

    run_command(
        "python scripts/train_transformer_dayhoff.py",
        "Step 5b: Train Transformer decoder (Dayhoff)",
        skip_steps, "decoder_train"
    )

    run_command(
        "python scripts/transformer_decode_dayhoff.py",
        "Step 6: Decode synthetic embeddings to amino acid sequences (Dayhoff)",
        skip_steps, "decode"
    )

    run_command(
        "python scripts/plot_tsne_domain_overlay_dayhoff.py",
        "Step 7a: Overlay real vs. generated embeddings via t-SNE (Dayhoff)",
        skip_steps, "tsne_overlay"
    )

    run_command(
        "python scripts/cosine_similarity_dayhoff.py",
        "Step 7b: Plot cosine similarity histogram (Dayhoff)",
        skip_steps, "cosine"
    )

    run_command(
        "python scripts/plot_entropy_identity_dayhoff.py",
        "Step 7c: Plot entropy vs. sequence identity (Dayhoff)",
        skip_steps, "entropy"
    )

    run_command(
        "python scripts/blastlocal_dayhoff.py",
        "Step 7d: Run local BLAST and summarize results (Dayhoff)",
        skip_steps, "blast"
    )

    # === Perplexity Analysis ===
    if "perplexity" in skip_steps:
        print("⏭️ Skipping: Step 7e - Perplexity Scoring Analysis (Dayhoff)")
    else:
        print("\n✅ Step 7e: Perplexity Scoring Analysis (Dayhoff)")
        run_command(
            "python scripts/perplexity_scoring.py",
            "Step 7e: Perplexity Scoring Analysis (Dayhoff)",
            skip_steps, "perplexity"
        )

    if "perplexity_plots" in skip_steps:
        print("⏭️ Skipping: Step 7f - Perplexity Plotting (Dayhoff)")
    else:
        print("\n✅ Step 7f: Perplexity Plotting (Dayhoff)")
        run_command(
            "python scripts/plot_perplexity_comparison.py",
            "Step 7f: Perplexity Plotting (Dayhoff)",
            skip_steps, "perplexity_plots"
        )

    # === Final HTML Report ===
    if "html" in skip_steps:
        print("⏭️ Skipping: Step 8 - Generate HTML Summary Report (Dayhoff)")
    else:
        print("\n✅ Step 8: Generate HTML Summary Report (Dayhoff)")
        run_command(
            "python scripts/generate_dayhoff_report.py",
            "Step 8: Generate HTML Summary Report (Dayhoff)",
            skip_steps, "html"
        )

    print("\n🎉 All steps in the EmbedDiff-Dayhoff pipeline completed successfully!")

if __name__ == "__main__":
    main()
