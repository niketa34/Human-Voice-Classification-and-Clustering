# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

ARTIFACTS_DIR = "artifacts"
EDA_DIR = os.path.join(ARTIFACTS_DIR, "eda")
DATA_PATH = "vocal_gender_features_new.csv"

st.set_page_config(page_title="Voice Classification & Clustering", layout="wide")

# Load essentials
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
    top_features = pd.read_csv(os.path.join(ARTIFACTS_DIR, "top_features.csv"))["feature"].tolist()

    # Raw data for EDA display if needed
    df = pd.read_csv(DATA_PATH)
    return scaler, model, top_features, df

scaler, model, top_features, df = load_artifacts()

# Pages
page = st.sidebar.selectbox("Navigate", ["Introduction", "EDA & Visualization", "Classification & Clustering"])

# ---------------------------
# Introduction Page
# ---------------------------
if page == "Introduction":
    st.title("Human Voice Classification & Clustering")
    st.markdown("""
    - Goal: Predict gender (classification) and group similar voices (clustering) using numeric audio features.
    - Dataset: Extracted features (spectral, pitch, MFCCs, energy) with label (male=1, female=0).
    - Pipeline:
      1. Data split (train/val/test)
      2. Preprocessing (scaling)
      3. EDA (distributions, correlations, PCA)
      4. Modeling (5 classifiers, 3 clustering methods)
      5. Evaluation (Accuracy, Precision, Recall, F1; Silhouette, Purity)
      6. Deployment (this Streamlit app)
    """)

    st.subheader("Artifacts")
    st.write("Saved models and EDA figures are in the 'artifacts' folder.")
    st.write("- scaler.pkl")
    st.write("- best_model.pkl")
    st.write("- top_features.csv")
    st.write("- EDA images under 'artifacts/eda'")

    st.subheader("Dataset preview")
    st.dataframe(df.head())

# ---------------------------
# EDA & Visualization Page
# ---------------------------
elif page == "EDA & Visualization":
    st.title("EDA & Visualization")

    eda_imgs = sorted([f for f in os.listdir(EDA_DIR) if f.endswith(".png")])
    if not eda_imgs:
        st.warning("No EDA images found. Run train_eval.py first.")
    else:
        # Display in a grid
        cols = st.columns(2)
        for i, img in enumerate(eda_imgs):
            with cols[i % 2]:
                st.image(os.path.join(EDA_DIR, img), caption=img, use_column_width=True)

    st.subheader("Dataset stats")
    st.write(df.describe())

# ---------------------------
# Classification & Clustering Page
# ---------------------------
else:
    st.title("Classification & Clustering")

    st.markdown("Enter numeric values for the selected top features to get predictions:")

    # Build inputs
    input_vals = []
    col1, col2 = st.columns(2)
    for i, feat in enumerate(top_features):
        container = col1 if i % 2 == 0 else col2
        with container:
            default_val = float(df[feat].median()) if feat in df.columns else 0.0
            val = st.number_input(f"{feat}", value=default_val, format="%.6f")
            input_vals.append(val)

    if st.button("Predict"):
        # Classification
        arr = np.array(input_vals).reshape(1, -1)

        # We need scaler for the full feature-space. Since we trained models on full features,
        # we’ll create a placeholder vector of full columns with provided top-features mapped.
        # Simplify: scale only the provided features using the same scaler fit on train subset.
        # Strict approach: re-scale via fitted scaler expects full order; here we project:
        # For practical demo, we re-fit a mini scaler on df[top_features] to scale input consistently.
        # Better: keep a separate scaler for top_features during training; but for demo this works.

        mini_scaler = StandardScaler()
        mini_scaler.fit(df[top_features].values)
        arr_scaled = mini_scaler.transform(arr)

        # Predict with a shallow model retrained on top_features? For demo, try model.predict on full scaler is not possible.
        # Workaround: we trained and saved the best model on full features. To support top_features-only input,
        # we’ll also fit a lightweight classifier on top_features here for interactive demo.
        # In production, retrain best model using only top_features and save; here we train quick RF on-the-fly.

        from sklearn.ensemble import RandomForestClassifier
        X_top = df[top_features].values
        y = df["label"].astype(int).values
        rf_demo = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_demo.fit(mini_scaler.transform(X_top), y)
        pred = rf_demo.predict(arr_scaled)
        proba = rf_demo.predict_proba(arr_scaled)[0][pred[0]]

        st.success(f"Classification prediction: {'Male' if pred[0]==1 else 'Female'} (confidence ~{proba:.2f})")

        # Clustering with top_features
        # KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(mini_scaler.transform(X_top))
        cl_kmeans = kmeans.predict(arr_scaled)

        # DBSCAN (predicting is not directly supported; we infer by nearest core point cluster if available)
        dbscan = DBSCAN(eps=2.0, min_samples=5)
        labels_db = dbscan.fit_predict(mini_scaler.transform(X_top))
        # Heuristic: if any neighbors within eps, assign that cluster
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=5).fit(mini_scaler.transform(X_top))
        distances, indices = nbrs.kneighbors(arr_scaled)
        neighbor_labels = labels_db[indices[0]]
        neighbor_labels = neighbor_labels[neighbor_labels != -1]
        cl_dbscan = neighbor_labels[0] if len(neighbor_labels) > 0 else -1

        # Agglomerative (predict requires assignment by nearest cluster centroid we compute)
        agg = AgglomerativeClustering(n_clusters=2)
        labels_agg = agg.fit_predict(mini_scaler.transform(X_top))
        # Compute cluster centroids for assignment
        centroids = []
        for c in np.unique(labels_agg):
            centroids.append(mini_scaler.transform(X_top)[labels_agg==c].mean(axis=0))
        centroids = np.vstack(centroids)
        from numpy.linalg import norm
        dists = norm(centroids - arr_scaled, axis=1)
        cl_agg = int(np.argmin(dists))

        st.info(f"Clustering assignments — KMeans: {cl_kmeans[0]}, DBSCAN: {cl_dbscan}, Agglomerative: {cl_agg}")

        # Simple visualization (2D PCA)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(mini_scaler.transform(X_top))
        inp_pca = pca.transform(arr_scaled)

        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", s=25, alpha=0.6, label="Dataset")
        ax.scatter(inp_pca[:,0], inp_pca[:,1], c="black", s=80, marker="X", label="Your input")
        ax.set_title("PCA projection (dataset vs your input)")
        ax.legend()
        st.pyplot(fig)

    st.caption("Note: For demo, prediction uses top 20 features. For production, retrain and save models specifically on these features for consistent scaling and inference.")