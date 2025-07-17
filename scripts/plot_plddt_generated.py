import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
METRICS_PATH = "esmfold_output/esmfold_metrics.csv"
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)
FIGURE_PATH = os.path.join(FIGURE_DIR, "plddt_boxplot_generated.png")

# === Load Data ===
df = pd.read_csv(METRICS_PATH)
df = df[df["Mean_pLDDT"].notnull()].copy()
df["Source"] = "Generated"

# === Plot ===
sns.set(style="whitegrid", font="Arial")
plt.figure(figsize=(6, 6))

ax = sns.boxplot(
    x="Source", y="Mean_pLDDT",
    data=df,
    palette=["skyblue"],
    width=0.4
)

# === Format ===
ax.set_title("Mean pLDDT of Generated Sequences", fontsize=14, fontweight="bold")
ax.set_xlabel("Source", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean pLDDT", fontsize=12, fontweight="bold")
ax.tick_params(labelsize=11)
plt.ylim(0, 100)
plt.tight_layout()

# === Save Figure ===
plt.savefig(FIGURE_PATH, dpi=300)
print(f"✅ Saved boxplot to: {FIGURE_PATH}")
