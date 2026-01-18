# train_eval.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)

import joblib

DATA_PATH = "vocal_gender_features_new.csv"
ARTIFACTS_DIR = "artifacts"
EDA_DIR = os.path.join(ARTIFACTS_DIR, "eda")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# -----------------------
# 1. Load dataset
# -----------------------
df = pd.read_csv(DATA_PATH)

# Expect 'label' column: 1 = male, 0 = female
assert "label" in df.columns, "Dataset must contain a 'label' column."

X = df.drop("label", axis=1)
y = df["label"].astype(int)

# -----------------------
# 2. Train/Val/Test split
# -----------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# -----------------------
# 3. Preprocessing
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

# -----------------------
# 4. EDA (10+ graphs)
# -----------------------
# 4.1 Basic info summary saved to artifacts
df.describe().to_csv(os.path.join(ARTIFACTS_DIR, "data_describe.csv"))

# Helper to save figures
def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, name), dpi=150)
    plt.close()

# 1) Label balance
plt.figure(figsize=(5,4))
sns.countplot(x=y)
plt.title("Label distribution (0=female, 1=male)")
save_fig("01_label_distribution.png")

# 2) Correlation heatmap of top features (to keep readable)
plt.figure(figsize=(12,10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
save_fig("02_correlation_heatmap.png")

# 3) Histograms of selected features
selected_hist = [
    "mean_pitch","max_pitch","std_pitch",
    "rms_energy","zero_crossing_rate",
    "mean_spectral_centroid","mean_spectral_bandwidth",
    "mean_spectral_flatness","mean_spectral_rolloff","energy_entropy"
]
for i, col in enumerate(selected_hist, start=1):
    if col in df.columns:
        plt.figure(figsize=(5,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram: {col}")
        save_fig(f"03_hist_{col}.png")

# 4) Boxplots for outlier patterns (subset)
selected_box = ["mean_pitch","std_pitch","rms_energy","mean_spectral_centroid","mean_spectral_bandwidth"]
for col in selected_box:
    if col in df.columns:
        plt.figure(figsize=(5,4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
        save_fig(f"04_box_{col}.png")

# 5) Pairplot for a small subset (to avoid heavy compute)
pair_cols = ["mean_pitch","rms_energy","zero_crossing_rate","mean_spectral_centroid","label"]
pair_cols = [c for c in pair_cols if c in df.columns]
if len(pair_cols) >= 3:
    pp = sns.pairplot(df[pair_cols], hue="label")
    pp.fig.suptitle("Pairplot of selected features", y=1.02)
    pp.savefig(os.path.join(EDA_DIR, "05_pairplot.png"))
    plt.close()

# 6) PCA scatter (2D) colored by label
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))  # fit on full X for EDA
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", s=20, alpha=0.8)
plt.title("PCA (2 components) colored by label")
plt.xlabel("PC1"); plt.ylabel("PC2")
save_fig("06_pca_scatter.png")

# 7) MFCC summary histograms (if present)
mfcc_means = [c for c in df.columns if c.startswith("mfcc_") and c.endswith("_mean")]
mfcc_stds = [c for c in df.columns if c.startswith("mfcc_") and c.endswith("_std")]

for col in mfcc_means[:4]:  # limit to 4 to control number
    plt.figure(figsize=(5,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"MFCC mean histogram: {col}")
    save_fig(f"07_mfcc_mean_hist_{col}.png")

for col in mfcc_stds[:4]:
    plt.figure(figsize=(5,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"MFCC std histogram: {col}")
    save_fig(f"08_mfcc_std_hist_{col}.png")

# 8) Feature vs label boxplots
feat_label_box = ["mean_pitch","rms_energy","zero_crossing_rate","mean_spectral_centroid"]
for col in feat_label_box:
    if col in df.columns:
        plt.figure(figsize=(5,4))
        sns.boxplot(x=y, y=df[col])
        plt.title(f"{col} by label")
        plt.xlabel("label")
        save_fig(f"09_box_by_label_{col}.png")

# 9) Density plot example
if "mean_pitch" in df.columns:
    plt.figure(figsize=(5,4))
    sns.kdeplot(data=df, x="mean_pitch", hue="label", fill=True, common_norm=False, alpha=0.4)
    plt.title("Density: mean_pitch by label")
    save_fig("10_density_mean_pitch.png")

# 10) Scatter relation example
if "mean_spectral_centroid" in df.columns and "rms_energy" in df.columns:
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=df["mean_spectral_centroid"], y=df["rms_energy"], hue=y, alpha=0.7)
    plt.title("Scatter: spectral centroid vs rms energy")
    save_fig("11_scatter_centroid_rms.png")

print(f"EDA complete. Figures saved to: {EDA_DIR}")

# -----------------------
# 5. Clustering (3 models)
# -----------------------
cluster_results = {}

# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X_train_scaled)
sil_kmeans = silhouette_score(X_train_scaled, labels_kmeans)
# Cluster purity (map clusters to labels by majority)
def cluster_purity(cluster_labels, true_labels):
    df_tmp = pd.DataFrame({"cluster": cluster_labels, "true": true_labels})
    purity_sum = 0
    for c in np.unique(cluster_labels):
        majority = df_tmp[df_tmp["cluster"]==c]["true"].value_counts().max()
        purity_sum += majority
    return purity_sum / len(df_tmp)

pur_kmeans = cluster_purity(labels_kmeans, y_train)
cluster_results["KMeans"] = {"silhouette": sil_kmeans, "purity": pur_kmeans}

# DBSCAN (parameters may need tuning; using a conservative default)
dbscan = DBSCAN(eps=2.0, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_train_scaled)
# Exclude noise (-1) for silhouette if all or many are noise; handle exceptions
try:
    valid_mask = labels_dbscan != -1
    if valid_mask.sum() > 1 and len(np.unique(labels_dbscan[valid_mask])) > 1:
        sil_dbscan = silhouette_score(X_train_scaled[valid_mask], labels_dbscan[valid_mask])
    else:
        sil_dbscan = np.nan
except Exception:
    sil_dbscan = np.nan
# Purity (ignore noise)
pur_dbscan = cluster_purity(labels_dbscan[labels_dbscan!=-1], y_train[labels_dbscan!=-1]) if np.any(labels_dbscan!=-1) else np.nan
cluster_results["DBSCAN"] = {"silhouette": sil_dbscan, "purity": pur_dbscan}

# Agglomerative
agg = AgglomerativeClustering(n_clusters=2)
labels_agg = agg.fit_predict(X_train_scaled)
sil_agg = silhouette_score(X_train_scaled, labels_agg)
pur_agg = cluster_purity(labels_agg, y_train)
cluster_results["Agglomerative"] = {"silhouette": sil_agg, "purity": pur_agg}

print("Clustering results (train):")
for k, v in cluster_results.items():
    print(f"{k}: silhouette={v['silhouette']:.4f} | purity={v['purity']:.4f}")

# -----------------------
# 6. Classification (5 models)
# -----------------------
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, max_iter=500, random_state=42)
}

val_scores = {}
reports = {}

for name, model in classifiers.items():
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred, zero_division=0)
    rec = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_val_pred)
    val_scores[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}
    reports[name] = classification_report(y_val, y_val_pred, zero_division=0)
    print(f"\n{name} (Validation): acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
    print("Confusion matrix:\n", cm)

# Choose best by F1
best_name = max(val_scores, key=lambda k: val_scores[k]["f1"])
best_model = classifiers[best_name]

print(f"\nBest model on validation: {best_name} | F1={val_scores[best_name]['f1']:.4f}")

# Final test evaluation
y_test_pred = best_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_cm = confusion_matrix(y_test, y_test_pred)

print(f"\n{best_name} (Test): acc={test_acc:.4f}, prec={test_prec:.4f}, rec={test_rec:.4f}, f1={test_f1:.4f}")
print("Test Confusion matrix:\n", test_cm)

# Save best model
joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "best_model.pkl"))

# Save feature names and top-20 selection (based on RF importance if available, fallback to variance)
try:
    rf_tmp = RandomForestClassifier(n_estimators=400, random_state=42)
    rf_tmp.fit(X_train_scaled, y_train)
    importances = rf_tmp.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    top_features = X.columns[top_idx].tolist()
except Exception:
    # fallback: highest variance
    variances = X_train.var().sort_values(ascending=False)
    top_features = variances.index.tolist()[:20]

pd.Series(top_features, name="feature").to_csv(os.path.join(ARTIFACTS_DIR, "top_features.csv"), index=False)

print(f"\nArtifacts saved to '{ARTIFACTS_DIR}':")
print("- scaler.pkl")
print("- best_model.pkl")
print("- top_features.csv")
print("- EDA figures in 'artifacts/eda'")