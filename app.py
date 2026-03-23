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

    # Select features
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

    fig1, ax1 = plt.subplots()
    ax1.scatter(pca_data[:,0], pca_data[:,1], c=df['kmeans'])
    ax1.set_title("KMeans Clusters (PCA)")
    st.pyplot(fig1)

    # -----------------------------
    # DBSCAN
    # -----------------------------
    st.subheader("🔴 DBSCAN Anomaly Detection")

    eps = st.slider("Select eps value", 0.3, 2.0, 0.8)

    db = DBSCAN(eps=eps, min_samples=5)
    df['dbscan'] = db.fit_predict(scaled_data)

    colors = np.where(df['dbscan'] == -1, 'red', 'blue')

    fig2, ax2 = plt.subplots()
    ax2.scatter(pca_data[:,0], pca_data[:,1], c=colors)
    ax2.set_title("DBSCAN Anomalies (Red)")
    st.pyplot(fig2)

    # -----------------------------
    # ANOMALY TABLE
    # -----------------------------
    st.subheader("🚨 Detected Anomalies")

    anomalies = df[df['dbscan'] == -1]
    st.write(anomalies.head())

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    st.subheader("📌 Insights")

    st.write("• Traffic patterns grouped into clusters (Low, Medium, High)")
    st.write("• Anomalies represent unusual traffic conditions")
    st.write("• Time plays a major role in traffic behavior")

else:
    st.write("Please upload a dataset to proceed.")
