import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("🚀 Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # -----------------------------
    # UNIVERSAL FEATURE SELECTION
    # -----------------------------
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        st.error("❌ Need at least 2 numeric columns")
        st.stop()

    features = numeric_df.copy()

    # Clean data (VERY IMPORTANT)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())

    # Ensure enough rows
    if len(features) < 10:
        st.error("❌ Not enough data")
        st.stop()

    st.write("DEBUG:", features.shape)

    # -----------------------------
    # SCALING + PCA
    # -----------------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔵 KMeans", "🔴 DBSCAN"])

    # -----------------------------
    # EDA
    # -----------------------------
    with tab1:

        st.subheader("📊 Feature Distribution")

        for col in features.columns[:3]:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(features[col], bins=30)
            ax.set_title(col)
            plt.tight_layout()
            st.pyplot(fig)

    # -----------------------------
    # KMEANS
    # -----------------------------
    with tab2:

        k = st.slider("Clusters", 2, 6, 3)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df['kmeans'] = kmeans.fit_predict(scaled_data)

        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(pca_data[:,0], pca_data[:,1], c=df['kmeans'])
        ax.set_title("KMeans")
        plt.tight_layout()
        st.pyplot(fig)

    # -----------------------------
    # DBSCAN (100% SAFE)
    # -----------------------------
    with tab3:

        eps = st.slider("eps", 0.1, 5.0, 1.5)
        min_samples = st.slider("min_samples", 2, 10, 3)

        try:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(scaled_data)

            df['dbscan'] = labels

            colors = np.where(labels == -1, 'red', 'blue')

            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(pca_data[:,0], pca_data[:,1], c=colors)
            ax.set_title("DBSCAN")
            plt.tight_layout()
            st.pyplot(fig)

            anomalies = df[df['dbscan'] == -1]

            st.write("🚨 Anomalies:", len(anomalies))
            st.dataframe(anomalies.head())

        except Exception as e:
            st.error(f"DBSCAN Failed: {e}")

else:
    st.info("Upload a dataset to begin")
