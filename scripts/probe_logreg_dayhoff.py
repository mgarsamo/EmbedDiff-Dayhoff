
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import os


# === Paths ===
EMBED_PATH = "embeddings/dayhoff_embeddings.npy"
FASTA_PATH = "data/curated_thioredoxin_reductase.fasta"
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# === Load embeddings ===
embeddings = np.load(EMBED_PATH)
print(f"[INFO] Embedding shape: {embeddings.shape}")

# === Load labels ===
labels = []
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    desc = record.description
    label = desc.split("[")[-1].split("]")[0].strip().lower()
    labels.append(label)
labels = pd.Series(labels, dtype="category")
label_ids = labels.cat.codes.values
print(f"[INFO] Number of classes: {len(labels.cat.categories)} | Labels: {list(labels.cat.categories)}")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, label_ids, test_size=0.2, stratify=label_ids, random_state=42
)


# === Train logistic regression probe ===
clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[RESULT] Logistic Regression accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=labels.cat.categories))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels.cat.categories,
            yticklabels=labels.cat.categories)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Logistic Regression Confusion Matrix (Dayhoff)")
plt.tight_layout()
cm_path = os.path.join(FIGURE_DIR, "logreg_confusion_matrix_dayhoff.png")
plt.savefig(cm_path, dpi=300)
print(f"[FIGURE] Confusion matrix saved to {cm_path}")
plt.close()

# === Accuracy Bar Plot ===
report = classification_report(y_test, y_pred, target_names=labels.cat.categories, output_dict=True)
accs = [report[c]['recall'] for c in labels.cat.categories]
plt.figure(figsize=(6, 4))
sns.barplot(x=list(labels.cat.categories), y=accs, palette="Set2")
plt.ylim(0, 1)
plt.ylabel("Recall (Accuracy)")
plt.xlabel("Class")
plt.title("Logistic Regression Per-Class Recall (Dayhoff)")
plt.tight_layout()
bar_path = os.path.join(FIGURE_DIR, "logreg_per_class_recall_dayhoff.png")
plt.savefig(bar_path, dpi=300)
print(f"[FIGURE] Per-class recall barplot saved to {bar_path}")
plt.close()
