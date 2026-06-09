
# 🚦 Traffic Pattern & Anomaly Detection

An interactive Streamlit application that analyzes traffic data using **Unsupervised Machine Learning** techniques to discover traffic patterns and detect unusual traffic events.

The application performs Exploratory Data Analysis (EDA), K-Means clustering, PCA-based visualization, and DBSCAN anomaly detection on traffic volume data and provides downloadable results for further analysis.

---

## 📌 Features

### 📊 Exploratory Data Analysis

* Daily Traffic Trend Analysis
* Average Traffic by Hour
* Traffic Distribution by Hour
* Temperature vs Traffic Analysis

### 🔵 K-Means Clustering

* Groups similar traffic patterns
* Interactive cluster selection
* PCA-based cluster visualization

### 🔴 DBSCAN Anomaly Detection

* Detects unusual traffic conditions
* Adjustable EPS parameter
* Highlights anomalies in red

### 📥 Export Results

* Download complete processed dataset
* Download detected anomalies separately

### 📈 Interactive Dashboard

* Upload custom traffic datasets
* Dynamic visualizations
* Real-time clustering analysis

---

## 🎯 Project Objective

The primary objective of this project is to:

* Identify hidden traffic patterns from unlabeled data
* Group traffic conditions into meaningful clusters
* Detect abnormal traffic behavior
* Analyze the impact of weather and time on traffic volume
* Provide an interactive platform for traffic data exploration

---

## 🗂 Dataset

The project uses the **Metro Interstate Traffic Volume Dataset** containing traffic and weather information.

### Important Features

| Feature        | Description            |
| -------------- | ---------------------- |
| traffic_volume | Traffic count          |
| temp           | Temperature            |
| rain_1h        | Rainfall in last hour  |
| snow_1h        | Snowfall in last hour  |
| clouds_all     | Cloud coverage         |
| date_time      | Timestamp              |
| hour           | Extracted hour feature |

---

## ⚙️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

---

## 🔬 Machine Learning Techniques

### K-Means Clustering

K-Means is used to group similar traffic patterns into clusters.

**Purpose**

* Identify low traffic conditions
* Identify medium traffic conditions
* Identify high traffic conditions

---

### PCA (Principal Component Analysis)

PCA is applied to reduce high-dimensional traffic data into two dimensions for visualization.

**Benefits**

* Reduces dimensionality
* Preserves important information
* Improves cluster visualization

---

### DBSCAN

DBSCAN is used to identify anomalies in traffic patterns.

**Purpose**

* Detect unusual traffic events
* Identify outliers
* Highlight abnormal traffic behavior

---

## 📊 Workflow

```text
Traffic Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Selection
       │
       ▼
Data Scaling
       │
 ┌─────┴─────┐
 ▼           ▼
K-Means    DBSCAN
 ▼           ▼
Clusters   Anomalies
       │
       ▼
PCA Visualization
       │
       ▼
Interactive Dashboard
```

---

## 📸 Application Preview

### Dashboard Components

* Upload Dataset
* Raw Data Preview
* Daily Traffic Trend
* Average Traffic by Hour
* Traffic Distribution Analysis
* Temperature vs Traffic
* K-Means Clustering Visualization
* DBSCAN Anomaly Detection
* Download Results

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Traffic-Pattern-And-Anomaly-Detection.git
cd Traffic-Pattern-And-Anomaly-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Traffic-Pattern-And-Anomaly-Detection
│
├── app.py
├── Metro_Interstate_Traffic_Volume.csv
├── requirements.txt
├── README.md
└── screenshots
```

---

## 📈 Key Insights Obtained

* Traffic patterns can be grouped into distinct clusters.
* Traffic volume changes significantly across different hours of the day.
* Peak traffic periods are observed during daytime hours.
* Weather conditions contribute to traffic variations.
* DBSCAN successfully identifies unusual traffic conditions as anomalies.
* Most traffic records belong to normal operating conditions.

---

## 🔮 Future Improvements

* Real-time traffic monitoring
* Live traffic anomaly alerts
* Traffic forecasting using Machine Learning
* Interactive Plotly dashboards
* Advanced clustering techniques
* Smart City integration

---

## 👨‍💻 Author

**Sree**

### Project Type

Unsupervised Machine Learning | Clustering | Anomaly Detection | Streamlit Dashboard

---

⭐ If you found this project useful, consider giving the repository a star.
