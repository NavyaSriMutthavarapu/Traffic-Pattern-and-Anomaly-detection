import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Traffic Analysis", layout="wide")
st.title("🚦 Traffic Pattern & Anomaly Detection")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Traffic Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
    df['hour'] = df['date_time'].dt.hour
    df['holiday'] = df['holiday'].fillna('None')

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔵 KMeans", "🔴 DBSCAN"])

    # =====================================================
    # 📊 EDA (FIXED)
    # =====================================================
    with tab1:

        # ✅ 1. Daily Trend (FIXED CLUTTER)
        st.subheader("📅 Daily Traffic Trend")

        daily = df.resample('D', on='date_time')['traffic_volume'].mean()

        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(daily.index, daily.values)
        ax.set_title("Daily Average Traffic")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        # ✅ 2. Hourly Pattern (BEST GRAPH)
        st.subheader("⏰ Average Traffic by Hour")

        hourly = df.groupby('hour')['traffic_volume'].mean()

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(hourly.index, hourly.values, marker='o')
        ax.set_title("Traffic Pattern by Hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Traffic Volume")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        # ✅ 3. Boxplot (VERY IMPORTANT)
        st.subheader("📦 Traffic Distribution by Hour")

        fig, ax = plt.subplots(figsize=(6,4))
        df.boxplot(column='traffic_volume', by='hour', ax=ax)
        plt.title("Traffic Distribution per Hour")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)

        # ✅ 4. Clean Scatter
        st.subheader("🌡️ Temperature vs Traffic")

        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(df['temp'], df['traffic_volume'], alpha=0.3)
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Traffic")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    # -----------------------------
    # ML PREP
    # -----------------------------
    features = df[['traffic_volume','temp','rain_1h','snow_1h','clouds_all','hour']].fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # =====================================================
    # 🔵 KMEANS
    # =====================================================
    with tab2:

        st.subheader("🔵 K-Means Clustering")

        k = st.slider("Select number of clusters", 2, 6, 3)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df['kmeans'] = kmeans.fit_predict(scaled_data)

        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(pca_data[:,0], pca_data[:,1], c=df['kmeans'])
        ax.set_title("KMeans Clusters")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📊 Cluster Count")
        st.bar_chart(df['kmeans'].value_counts())

        # DOWNLOAD
        csv_kmeans = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download KMeans Results", csv_kmeans, "kmeans.csv")

    # =====================================================
    # 🔴 DBSCAN
    # =====================================================
    with tab3:

        st.subheader("🔴 DBSCAN Anomaly Detection")

        col1, col2 = st.columns(2)

        with col1:
            eps = st.slider("eps", 0.1, 5.0, 1.5)

        with col2:
            min_samples = st.slider("min_samples", 2, 10, 3)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        df['dbscan'] = db.fit_predict(scaled_data)

        colors = np.where(df['dbscan'] == -1, 'red', 'blue')

        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(pca_data[:,0], pca_data[:,1], c=colors)
        ax.set_title("DBSCAN Result")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        anomalies = df[df['dbscan'] == -1]

        st.subheader("🚨 Detected Anomalies")
        st.write(f"Total anomalies: {len(anomalies)}")
        st.dataframe(anomalies.head())

        # DOWNLOAD OPTIONS
        csv_full = df.to_csv(index=False).encode('utf-8')
        csv_anomaly = anomalies.to_csv(index=False).encode('utf-8')

        st.download_button("⬇️ Download Full Data", csv_full, "full_data.csv")
        st.download_button("⬇️ Download Anomalies", csv_anomaly, "anomalies.csv")

else:
    st.info("📂 Please upload a dataset to start analysis")