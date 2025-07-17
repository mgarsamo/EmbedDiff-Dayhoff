# scripts/select_near_clusters.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import os

# === Load data ===
tsne_coords = np.load("embeddings/tsne_coords.npy")  # shape: (n_total, 2)
colors = np.load("embeddings/tsne_labels.npy")       # e.g., ["kinase", ..., "Generated", "Generated"]
embeddings = np.load("embeddings/sampled_embeddings.npy")  # shape: (n_generated, 1280)

# === Extract real class clusters ===
real_classes = ["kinase", "beta_lactamase", "serine_protease"]
class_centroids = {}

for cls in real_classes:
    class_points = tsne_coords[np.array(colors) == cls]
    class_centroids[cls] = class_points.mean(axis=0)

# === Extract generated points ===
gen_mask = np.array(colors) == "Generated"
gen_coords = tsne_coords[gen_mask]
gen_embeddings = embeddings  # these are already aligned

# === Find generated points near each real cluster ===
RADIUS = 20  # adjust this if you want more or fewer samples
selected_embeddings = []

for cls, center in class_centroids.items():
    dists = euclidean_distances(gen_coords, center.reshape(1, -1)).flatten()
    near_mask = dists < RADIUS
    selected = gen_embeddings[near_mask]
    selected_embeddings.append(selected)
    print(f"✅ {cls}: Selected {selected.shape[0]} embeddings near cluster")

# === Save the filtered embeddings ===
final_selected = np.concatenate(selected_embeddings, axis=0)
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/filtered_embeddings.npy", final_selected)
print(f"\n🎯 Saved filtered embeddings to embeddings/filtered_embeddings.npy with shape: {final_selected.shape}")
