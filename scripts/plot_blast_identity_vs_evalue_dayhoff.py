# scripts/plot_blast_identity_vs_evalue_dayhoff.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set(style="white", context="talk", font_scale=1.2)

# === Paths ===
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "blast_results" / "blast_summary_local_dayhoff.csv"
output_path = project_root / "figures"
output_path.mkdir(parents=True, exist_ok=True)

print(f"📄 Looking for Dayhoff BLAST CSV at: {file_path}")

# === Load CSV ===
df = pd.read_csv(file_path)
print("🧠 Columns in CSV:", df.columns.tolist())

# === Clean column names ===
df.columns = df.columns.str.strip()

# === Clean and convert E-value column ===
df["Top_Hit_1_E-value"] = df["Top_Hit_1_E-value"].astype(str).str.replace(r"[^0-9eE.-]", "", regex=True)
df["Top_Hit_1_E-value"] = pd.to_numeric(df["Top_Hit_1_E-value"], errors="coerce")

# === Drop rows with missing values in required columns ===
df = df.dropna(subset=["Top_Hit_1_Identity(%)", "Top_Hit_1_E-value"])

# === Plot ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Top_Hit_1_Identity(%)",
    y=-np.log10(df["Top_Hit_1_E-value"]),
    data=df,
    s=60,
    color="darkslategray",
    edgecolor="black",
    linewidth=0.5,
    alpha=0.8
)

plt.xlabel("Top Hit Identity (%)", labelpad=10, weight='bold')
plt.ylabel(r"$-\log_{10}(\mathrm{E\text{-}value})$", labelpad=10, weight='bold')
sns.despine()
plt.tight_layout()

# === Save Figure ===
figure_file = output_path / "fig4a_identity_vs_evalue_dayhoff.png"
plt.savefig(figure_file, dpi=300)
plt.close()

print(f"✅ Dayhoff BLAST identity vs E-value figure saved to: {figure_file}")
