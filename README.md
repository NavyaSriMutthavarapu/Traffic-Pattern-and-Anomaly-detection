# 🚦 Traffic Pattern & Anomaly Detection
📌 Project Overview

Traffic management is an important aspect of smart cities and transportation systems. Understanding traffic behavior and identifying unusual traffic conditions can help in better traffic planning, congestion management, and anomaly detection.

This project uses Unsupervised Machine Learning techniques to analyze traffic patterns from the Metro Interstate Traffic Volume Dataset. The system groups similar traffic conditions using clustering algorithms and detects unusual traffic events using anomaly detection techniques.

An interactive Streamlit web application is developed to allow users to upload traffic datasets, visualize traffic trends, perform clustering, detect anomalies, and download results.

🎯 Objectives
Analyze traffic patterns using historical traffic data.
Identify Low, Medium, and High traffic conditions.
Detect unusual traffic events (anomalies).
Visualize traffic behavior using exploratory data analysis.
Provide an interactive dashboard for users.
Enable downloading of processed results.
📂 Dataset

Dataset: Metro Interstate Traffic Volume Dataset

Features Used
Feature	Description
traffic_volume	Number of vehicles passing
temp	Temperature
rain_1h	Rainfall in the last hour
snow_1h	Snowfall in the last hour
clouds_all	Cloud coverage percentage
date_time	Date and time
hour	Extracted hour feature
⚙️ Technologies Used
Python
Streamlit
Pandas
NumPy
Matplotlib
Scikit-Learn
🔍 Methodology
1. Data Preprocessing
Loaded traffic dataset
Converted date_time into datetime format
Extracted hourly information
Handled missing values
Selected important traffic and weather-related features
Standardized features using StandardScaler
2. Exploratory Data Analysis (EDA)

The following visualizations were performed:

📅 Daily Traffic Trend

Shows how average traffic changes over time.

⏰ Average Traffic by Hour

Helps identify peak traffic hours and low traffic periods.

📦 Traffic Distribution by Hour

Visualizes traffic variability throughout the day.

🌡️ Temperature vs Traffic

Examines the relationship between temperature and traffic volume.

3. K-Means Clustering

K-Means clustering was used to group similar traffic conditions.

Purpose:

Identify traffic patterns.
Categorize traffic into clusters such as:
Low Traffic
Medium Traffic
High Traffic

Visualization:

PCA was applied to reduce dimensions.
Cluster assignments were visualized in 2D space.
4. Principal Component Analysis (PCA)

PCA was used for dimensionality reduction.

Why PCA?

Original dataset contains multiple features.
PCA transforms them into two principal components.
Makes visualization easier while preserving most information.
5. DBSCAN Anomaly Detection

DBSCAN was applied to identify unusual traffic conditions.

Purpose:

Detect rare traffic events.
Identify outliers that do not belong to any cluster.

Output Labels:

Label	Meaning
0,1,2...	Normal Cluster
-1	Anomaly / Outlier

Anomalies are highlighted in red within the visualization.

📊 Results
Traffic Insights

✔ Traffic patterns can be grouped into distinct clusters.

✔ Traffic volume changes significantly throughout the day.

✔ Peak traffic hours were observed during daytime.

✔ Most records belong to normal traffic conditions.

✔ A small number of unusual traffic events were successfully detected using DBSCAN.

🖥️ Streamlit Application Features
📁 Dataset Upload

Upload a CSV traffic dataset.

📊 Raw Data Preview

View dataset contents.

📈 Traffic Trend Analysis
Daily Traffic Trend
Hourly Traffic Pattern
Traffic Distribution
🔵 K-Means Clustering
Select number of clusters
Visualize cluster groups
🔴 DBSCAN Anomaly Detection
Adjustable EPS parameter
Detect and visualize anomalies
🚨 Anomaly Table

View detected anomalies.

⬇️ Download Results
Download complete processed dataset
Download anomaly records only
📌 Insights Section

Summarized observations generated from analysis.

📸 Application Screenshots
Home Page
Dataset Upload
Raw Data Preview
Exploratory Data Analysis
Daily Traffic Trend
Hourly Traffic Pattern
Traffic Distribution
Clustering Visualization
K-Means Clusters using PCA
Anomaly Detection
DBSCAN Visualization
Detected Anomalies Table
▶️ How to Run
Clone Repository
git clone https://github.com/your-username/traffic-pattern-and-anomaly-detection.git
cd traffic-pattern-and-anomaly-detection
Install Dependencies
pip install -r requirements.txt
Run Streamlit App
streamlit run app.py
📁 Project Structure
Traffic-Pattern-And-Anomaly-Detection/
│
├── app.py
├── Metro_Interstate_Traffic_Volume.csv
├── requirements.txt
├── README.md
└── screenshots/
🔮 Future Enhancements
Real-time traffic monitoring
Interactive Plotly visualizations
Traffic forecasting using Machine Learning
Weather impact analysis
Advanced anomaly detection techniques
Smart city traffic management integration
👨‍💻 Author

Sree

Project: Traffic Pattern & Anomaly Detection Using Unsupervised Learning

⭐ Key Learning

This project demonstrates how Unsupervised Learning can discover hidden traffic patterns and detect unusual traffic events without requiring labeled data, making it valuable for traffic monitoring and intelligent transportation systems.
