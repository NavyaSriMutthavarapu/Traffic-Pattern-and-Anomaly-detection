import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# -----------------------------
# TITLE
# -----------------------------
st.title("🚦 Traffic Pattern & Anomaly Detection")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Traffic Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.write(df.head())

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['holiday'] = df['holiday'].fillna('None')

    # -----------------------------
    # 📊 EDA SECTION (ADDED)
    # -----------------------------
    st.header("📊 Exploratory Data Analysis")

    # 1. Daily Trend (clean)
    st.subheader("📅 Daily Traffic Trend")
    daily = df.resample('D', on='date_time')['traffic_volume'].mean()

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(daily.index, daily.values)
    ax.set_title("Daily Average Traffic")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # 2. Hourly Pattern (best graph)
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

    # 3. Boxplot
    st.subheader("📦 Traffic Distribution by Hour")

    fig, ax = plt.subplots(figsize=(6,4))
    df.boxplot(column='traffic_volume', by='hour', ax=ax)
    plt.title("Traffic Distribution per Hour")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig)

    # 4. Scatter (no clutter)
    st.subheader("🌡️ Temperature vs Traffic")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df['temp'], df['traffic_volume'], alpha=0.3)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Traffic Volume")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    features = df[['traffic_volume','temp','rain_1h','snow_1h','clouds_all','hour']]

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    # -----------------------------
    # KMEANS
    # -----------------------------
    st.subheader("🔵 K-Means Clustering")

    k = st.slider("Select number of clusters", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df['kmeans'] = kmeans.fit_predict(scaled_data)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    fig1, ax1 = plt.subplots(figsize=(6,5))
    ax1.scatter(pca_data[:,0], pca_data[:,1], c=df['kmeans'])
    ax1.set_title("KMeans Clusters (PCA)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig1)

    # -----------------------------
    # DBSCAN
    # -----------------------------
    st.subheader("🔴 DBSCAN Anomaly Detection")

    eps = st.slider("Select eps value", 0.3, 2.0, 0.8)

    db = DBSCAN(eps=eps, min_samples=5)
    df['dbscan'] = db.fit_predict(scaled_data)

    colors = np.where(df['dbscan'] == -1, 'red', 'blue')

    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.scatter(pca_data[:,0], pca_data[:,1], c=colors)
    ax2.set_title("DBSCAN Anomalies (Red)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig2)

    # -----------------------------
    # ANOMALY TABLE
    # -----------------------------
    st.subheader("🚨 Detected Anomalies")

    anomalies = df[df['dbscan'] == -1]
    st.write(f"Total anomalies: {len(anomalies)}")
    st.write(anomalies.head())
    

    # -----------------------------
    # DOWNLOAD OPTIONS (ADDED)
    # -----------------------------
    st.subheader("⬇️ Download Results")

    csv_full = df.to_csv(index=False).encode('utf-8')
    csv_anomaly = anomalies.to_csv(index=False).encode('utf-8')

    st.download_button("Download Full Dataset", csv_full, "full_data.csv")
    st.download_button("Download Anomalies Only", csv_anomaly, "anomalies.csv")

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    st.subheader("📌 Insights")

    st.write("• Traffic patterns grouped into clusters (Low, Medium, High)")
    st.write("• Anomalies represent unusual traffic conditions")
    st.write("• Time plays a major role in traffic behavior")

else:
    st.write("Please upload a dataset to proceed.")
