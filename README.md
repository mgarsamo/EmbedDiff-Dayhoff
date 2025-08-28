

# 🧬 EmbedDiff-Dayhoff: Protein Sequence Generation via Latent Diffusion

A comprehensive implementation of the EmbedDiff pipeline for generating novel protein sequences using Dayhoff embeddings and latent diffusion models.

## 🔬 About This Study

**EmbedDiff-Dayhoff** is an **ablation study** that extends the original [EmbedDiff pipeline](https://github.com/mgarsamo/EmbedDiff) by swapping the embedding backbone from **ESM-2** to **Microsoft Dayhoff Atlas**. 

This repository isolates the key question: **How do Dayhoff embeddings (trained on clustered UniRef) affect de novo sequence generation compared to the ESM-2 baseline?**

### 🔄 What's Different in This Ablation?

- **Embedding Backbone**: ESM-2 → **Dayhoff Atlas** (default: `microsoft/Dayhoff-3b-UR90`)
- **End-to-End Dayhoff Scripts**: All pipeline steps are Dayhoff-specific (`*_dayhoff.py`)
- **Dimension-Agnostic**: Auto-detects embedding dimensions from saved `.npy` files
- **Mamba Compatibility**: Handles Jamba/Mamba architecture with CPU support

> 💾 **Model size**: `Dayhoff-3b-UR90` is ~12 GB (3 shards). Use `--batch_size` conservatively on CPU/MPS.  
> ⚙️ **Mamba kernels**: Loaded with `use_mamba_kernels=False` to avoid CUDA-only kernel warnings on Mac.

### 📚 Related Work

- **Original EmbedDiff**: [ESM-2 + Latent Diffusion Pipeline](https://github.com/mgarsamo/EmbedDiff)
- **This Study**: Dayhoff + Latent Diffusion Pipeline (current repository)
- **Comparison**: Evaluate embedding model effects on protein generation quality

---

## 📋 Overview

This repository implements a complete protein generation pipeline that:
1. **Generates protein embeddings** using the Microsoft Dayhoff protein language model
2. **Trains a latent diffusion model** to learn the distribution of protein embeddings
3. **Samples synthetic embeddings** from the learned distribution
4. **Reconstructs protein sequences** using a transformer decoder
5. **Evaluates generated sequences** through comprehensive analysis and visualization

## 🚀 Key Features

- **Dayhoff Embeddings**: Uses `microsoft/Dayhoff-3b-UR90` for high-quality protein representations
- **Latent Diffusion**: Implements cosine noise scheduling with improved normalization
- **Transformer Decoder**: Reconstructs sequences from embeddings with high fidelity
- **Comprehensive Analysis**: t-SNE visualization, similarity analysis, quality metrics, and BLAST evaluation
- **Professional Reporting**: Generates HTML reports with all visualizations and results

## 📊 Results Summary

Our pipeline successfully:
- ✅ **Generated 240 high-quality synthetic protein sequences**
- ✅ **Achieved 32-68% sequence identity** (most around 55-65%)
- ✅ **Trained diffusion model** with cosine noise schedule and [-1,1] normalization
- ✅ **Trained transformer decoder** with 15% loss improvement over 34 epochs
- ✅ **Maintained biological plausibility** through domain-aware embedding generation

## 📋 View Complete Results

**📊 [View Full HTML Report](embeddiff_dayhoff_summary_report.html)** - Comprehensive analysis with all 13 figures, metrics, and downloadable data

The HTML report contains:
- All generated visualizations and analysis plots
- Performance metrics and training curves
- Sequence quality assessments and BLAST results
- Downloadable FASTA files and CSV data
- Professional presentation of all pipeline outputs

## 🏗️ Architecture

```
Real Protein Sequences → Dayhoff Embeddings → Latent Diffusion Model → Synthetic Embeddings → Transformer Decoder → Novel Protein Sequences
```

### Core Components

1. **Dayhoff Embedder** (`utils/dayhoff_embedder.py`)
   - Generates 1280-dimensional embeddings using Microsoft's Dayhoff-3B model
   - Handles Jamba/Mamba architecture with CPU compatibility
   - Supports batch processing and custom device selection

2. **Latent Diffusion Model** (`models/latent_diffusion.py`)
   - MLP-based noise predictor with dynamic timestep scaling
   - Cosine beta schedule for smooth noise addition
   - Configurable timesteps (default: 1000) and learning parameters

3. **Transformer Decoder** (`models/decoder_transformer.py`)
   - 4-layer transformer architecture with 512 embedding dimensions
   - Trained to reconstruct protein sequences from embeddings
   - Early stopping and model checkpointing

## 📁 Repository Structure

```
EmbedDiff_Dayhoff/
├── data/                           # Input/output data files
│   ├── curated_thioredoxin_reductase.fasta
│   ├── thioredoxin_reductase.fasta
│   ├── decoded_embeddiff_dayhoff.fasta
│   └── blast_results/              # BLAST analysis results
├── embeddings/                     # Generated embeddings
│   ├── dayhoff_embeddings.npy
│   └── sampled_dayhoff_embeddings.npy
├── figures/                        # All generated visualizations
│   ├── fig_tsne_by_domain_dayhoff.png
│   ├── fig2b_loss_dayhoff.png
│   ├── fig3a_generated_tsne_dayhoff.png
│   ├── fig5a_decoder_loss_dayhoff.png
│   ├── fig5a_real_real_cosine_dayhoff.png
│   ├── fig5b_gen_gen_cosine_dayhoff.png
│   ├── fig5c_real_gen_cosine_dayhoff.png
│   ├── fig5b_identity_histogram_dayhoff.png
│   ├── fig5c_entropy_scatter_dayhoff.png
│   ├── fig5d_all_histograms_dayhoff.png
│   ├── fig5f_tsne_domain_overlay_dayhoff.png
│   ├── logreg_per_class_recall_dayhoff.png
│   └── logreg_confusion_matrix_dayhoff.png
├── models/                         # Model architectures
│   ├── latent_diffusion.py
│   └── decoder_transformer.py
├── scripts/                        # Pipeline execution scripts
│   ├── run_embeddiff_pipeline_dayhoff.py
│   ├── generate_dayhoff_embeddings.py
│   ├── train_embeddiff_dayhoff.py
│   ├── sample_embeddings_dayhoff.py
│   ├── build_decoder_dataset_dayhoff.py
│   ├── train_transformer_dayhoff.py
│   ├── transformer_decode_dayhoff.py
│   ├── plot_tsne_by_domain_dayhoff.py
│   ├── plot_tsne_domain_overlay_dayhoff.py
│   ├── cosine_similarity_dayhoff.py
│   ├── plot_entropy_identity_dayhoff.py
│   ├── plot_blast_identity_vs_evalue_dayhoff.py
│   ├── blastlocal_dayhoff.py
│   ├── generate_dayhoff_report.py
│   └── train_transformer_dayhoff.py
├── utils/                          # Utility functions
│   ├── dayhoff_embedder.py
│   └── esm_embedder.py
├── checkpoints/                    # Trained model checkpoints
├── notebooks/                      # Jupyter notebooks for exploration
├── requirements.txt                # Python dependencies
└── embeddiff_dayhoff_summary_report.html  # Comprehensive results report
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.3.1+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mgarsamo/EmbedDiff-Dayhoff.git
   cd EmbedDiff-Dayhoff
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Get Started Quickly

### Option 1: Run Complete Pipeline (Recommended)

Execute the entire EmbedDiff-Dayhoff pipeline with one command:

```bash
# Activate your environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the complete pipeline
python run_embeddiff_pipeline_dayhoff.py
```

**What this does:**
- ✅ Generates Dayhoff embeddings from your protein sequences
- ✅ Trains the latent diffusion model
- ✅ Samples synthetic embeddings
- ✅ Trains the transformer decoder
- ✅ Decodes sequences and runs all analyses
- ✅ Generates comprehensive HTML report

**Expected time:** 2-4 hours depending on your hardware

### Option 2: Run with Custom Skip Options

Skip specific steps if you want to resume from a certain point:

```bash
# Skip embedding generation (if you already have embeddings)
python run_embeddiff_pipeline_dayhoff.py --skip dayhoff

# Skip BLAST analysis (if you don't have BLAST+ installed)
python run_embeddiff_pipeline_dayhoff.py --skip blast

# Skip multiple steps
python run_embeddiff_pipeline_dayhoff.py --skip dayhoff tsne diffusion

# Available skip options: dayhoff, tsne, diffusion, sample, decoder_data, decoder_train, decode, tsne_overlay, cosine, entropy, blast, html
```

### Option 3: Step-by-Step Execution

Run individual components for debugging or customization:

```bash
# Step 1: Generate Dayhoff embeddings
python utils/dayhoff_embedder.py --input data/curated_thioredoxin_reductase.fasta --output embeddings/dayhoff_embeddings.npy

# Step 2: Visualize real embeddings
python scripts/plot_tsne_by_domain_dayhoff.py

# Step 3: Train diffusion model
python scripts/train_embeddiff_dayhoff.py

# Step 4: Sample synthetic embeddings
python scripts/sample_embeddings_dayhoff.py

# Step 5: Build decoder dataset
python scripts/build_decoder_dataset_dayhoff.py

# Step 6: Train transformer decoder
python scripts/train_transformer_dayhoff.py

# Step 7: Decode to sequences
python scripts/transformer_decode_dayhoff.py

# Step 8: Generate HTML report
python scripts/generate_dayhoff_report.py
```

### 🎯 Quick Test Run

Want to test the pipeline quickly? Use a smaller dataset:

```bash
# Create a small test dataset
head -20 data/curated_thioredoxin_reductase.fasta > data/test_dataset.fasta

# Run pipeline on test data
python utils/dayhoff_embedder.py --input data/test_dataset.fasta --output embeddings/test_embeddings.npy
python scripts/plot_tsne_by_domain_dayhoff.py
```

### 📊 View Results

After running the pipeline, view your results:

```bash
# Open the comprehensive HTML report
open embeddiff_dayhoff_summary_report.html

# Or view individual figures
ls figures/
```

---

## 📊 Expected Outputs

After running the complete pipeline, you'll have:

### 🗂️ Generated Files

- **`embeddings/dayhoff_embeddings.npy`** - Real protein embeddings (1280D)
- **`embeddings/sampled_dayhoff_embeddings.npy`** - Synthetic embeddings
- **`data/decoded_embeddiff_dayhoff.fasta`** - 240 generated protein sequences
- **`checkpoints/`** - Trained model checkpoints
- **`figures/`** - 13 comprehensive analysis plots
- **`embeddiff_dayhoff_summary_report.html`** - Complete results report

### 📈 Key Metrics You'll See

- **Sequence Generation**: 240 high-quality synthetic proteins
- **Identity Range**: 32-68% similarity to real sequences
- **Classification Accuracy**: 92% domain prediction performance
- **Training Progress**: Loss curves and convergence metrics
- **Quality Validation**: Entropy, identity, and BLAST analysis

### 🔍 What Each Figure Shows

1. **Domain Separation** - How well Dayhoff separates biological domains
2. **Classification Performance** - Logistic regression accuracy metrics
3. **Diffusion Training** - Model convergence and loss reduction
4. **Generated Embeddings** - Synthetic vs. real embedding comparison
5. **Sequence Quality** - Identity distributions and entropy analysis
6. **Similarity Analysis** - Cosine similarity between sequence types

---

## 📈 Key Results & Visualizations

### 1. Domain Separation (Figure 1)
- Clear separation of bacteria, fungi, and archaea in embedding space
- Demonstrates Dayhoff model's ability to capture biological relationships

### 2. Classification Performance (Figures 2-3)
- **92% overall accuracy** in domain classification
- Strong per-class recall: Archaea (89%), Bacteria (84%), Fungi (99%)

### 3. Diffusion Training (Figure 4)
- Successful training with cosine noise schedule
- Loss reduction from 12.66 to 10.79 (15% improvement)

### 4. Generated Embeddings (Figure 5)
- Synthetic embeddings overlap with real protein distributions
- Maintains biological plausibility across domains

### 5. Sequence Quality (Figures 10-11)
- **Identity range**: 32-68% (most around 55-65%)
- **Entropy threshold**: All sequences above 2.8 Shannon entropy
- **Quality filtering**: Comprehensive validation of generated sequences

### 6. Similarity Analysis (Figures 7-9)
- High cosine similarity between real and generated sequences
- Generated sequences show internal coherence and diversity

## 🔬 Technical Details

### Model Specifications

- **Dayhoff Model**: `microsoft/Dayhoff-3b-UR90` (3B parameters, 1280D embeddings)
- **Diffusion Model**: MLP noise predictor with 1000 timesteps
- **Transformer Decoder**: 4 layers, 512 embedding dims, 8 attention heads
- **Training**: Adam optimizer, learning rate 1e-4, batch size 32

### Key Innovations

1. **Cosine Noise Schedule**: Smoother noise addition for better training stability
2. **[-1,1] Normalization**: Improved embedding scaling for diffusion models
3. **Dynamic Timestep Scaling**: Adaptive normalization based on total timesteps
4. **CPU Compatibility**: Mamba kernel disabling for broad accessibility

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Generated Sequences** | 240 | High-quality synthetic proteins |
| **Sequence Identity** | 32-68% | Range of similarity to real sequences |
| **Classification Accuracy** | 92% | Domain prediction performance |
| **Training Epochs** | 34 | Transformer decoder training |
| **Loss Improvement** | 15% | Diffusion model training progress |

## 🎯 Applications

- **Drug Discovery**: Generate novel protein therapeutics
- **Protein Engineering**: Design proteins with specific functions
- **Evolutionary Studies**: Understand protein sequence space
- **Bioinformatics Research**: Explore protein sequence relationships

## 🔬 Ablation Study Methodology

### Comparing Dayhoff vs ESM-2 Results

This repository enables direct comparison with the [original EmbedDiff ESM-2 pipeline](https://github.com/mgarsamo/EmbedDiff):

1. **Run both pipelines** on the same input dataset
2. **Compare key metrics**:
   - Sequence identity distributions
   - Training loss curves
   - t-SNE embedding distributions
   - Classification performance
   - BLAST validation results

3. **Evaluate differences** in:
   - **Embedding quality**: Domain separation and biological relationships
   - **Generation diversity**: Novelty vs. biological plausibility
   - **Training stability**: Convergence and loss patterns
   - **Computational efficiency**: Model size and inference speed

### Key Research Questions

- **Does Dayhoff's UniRef clustering improve domain-aware generation?**
- **How do 1280D Dayhoff embeddings compare to ESM-2's 1280D?**
- **Which embedding model produces more biologically plausible sequences?**
- **What are the trade-offs between model size and generation quality?**

### Reproducing the Comparison

```bash
# ESM-2 baseline (from original repository)
git clone https://github.com/mgarsamo/EmbedDiff.git
cd EmbedDiff
python run_embeddiff_pipeline.py

# Dayhoff ablation (current repository)
git clone https://github.com/mgarsamo/EmbedDiff-Dayhoff.git
cd EmbedDiff-Dayhoff
python run_embeddiff_pipeline_dayhoff.py

# Compare results in respective HTML reports
```

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Submit issues and feature requests
- Contribute code improvements
- Share research applications and results
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft Research** for the Dayhoff protein language models
- **Hugging Face** for the transformers library
- **PyTorch** team for the deep learning framework
- **Bioinformatics community** for protein analysis tools

## 📞 Contact

- **GitHub**: [@mgarsamo](https://github.com/mgarsamo)
- **Repository**: [EmbedDiff-Dayhoff](https://github.com/mgarsamo/EmbedDiff-Dayhoff)

---

**Last Updated**: August 28, 2025  
**Pipeline Status**: ✅ Complete and Fully Functional  
**Results**: Available in `embeddiff_dayhoff_summary_report.html`


