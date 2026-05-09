# Grocery Demand Forecasting with Anomaly Alerts

## Project Overview

This project predicts grocery demand using machine learning forecasting models and detects anomalies using statistical and density-based methods.

The system combines:

- CatBoost Regressor
- LightGBM Regressor
- Dynamic Weighted Ensemble
- Local Outlier Factor (LOF)
- Z-Score Validation

The project also includes a professional Streamlit dashboard for visualization and monitoring.

---

## Features

- Grocery demand forecasting
- Dynamic weighted ensemble prediction
- Residual analysis
- LOF anomaly detection
- Z-score validation layer
- Interactive Streamlit dashboard
- Forecast confidence band
- Downloadable forecast reports

---

## Dashboard Preview

![Dashboard](assets/dashboard.png)

---

## Forecast Visualization

![Forecast Plot](assets/forecast_plot.png)

---

## Anomaly Detection

![Anomaly Plot](assets/anomaly_plot.png)

---

## Project Architecture

![Architecture](assets/architecture.png)

---

## Dataset Information

The dataset contains:

- Store information
- Product information
- Inventory details
- Sales records
- Weather conditions
- Promotion status
- Competitor pricing
- Seasonality information
- Demand values

---

## Machine Learning Pipeline

```text
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Baseline Forecasting
        ↓
CatBoost Training
        ↓
LightGBM Training
        ↓
Dynamic Ensemble Forecasting
        ↓
Residual Analysis
        ↓
LOF Anomaly Detection
        ↓
Z-score Validation
        ↓
Final Anomaly Alerts
        ↓
Streamlit Dashboard
````

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPO_LINK
```

Move into the project folder:

```bash
cd grocery-demand-forecast
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Model Performance

The ensemble forecasting model achieved:

* High R² score
* Low RMSE
* Low MAE
* Strong forecasting accuracy

---

## Technologies Used

* Python
* Streamlit
* Plotly
* Pandas
* NumPy
* CatBoost
* LightGBM
* Scikit-learn

---

## Future Improvements

* Real-time forecasting
* API integration
* Cloud deployment
* Deep learning forecasting
* Automated retraining pipeline

---

## Author

Md. Nafijul Islam

```
```
=======
# Grocery-Demand-Forecasting-with-Anomaly-Alerts
>>>>>>> e4e2d66fe7976e76fb09c4d597774f3e120bc2e0
