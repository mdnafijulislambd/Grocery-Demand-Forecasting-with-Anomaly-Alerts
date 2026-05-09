# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import joblib
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score
# )


# st.set_page_config(
#     page_title="Grocery Demand Forecasting",
#     page_icon="🛒",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# # =========================
# # Load Models
# # =========================

# @st.cache_resource

# def load_models():

#     cat_model = joblib.load(
#         "models/catboost_model.pkl"
#     )

#     lgb_model = joblib.load(
#         "models/lightgbm_model.pkl"
#     )

#     label_encoders = joblib.load(
#         "models/label_encoders.pkl"
#     )

#     return cat_model, lgb_model, label_encoders


# cat_model, lgb_model, label_encoders = load_models()


# # =========================
# # Load Dataset
# # =========================

# @st.cache_data

# def load_data():

#     df = pd.read_csv(
#         "data/forecast_dashboard_data.csv"
#     )

#     return df


# prediction_df = load_data()


# # =========================
# # Header
# # =========================

# st.title("🛒 Grocery Demand Forecasting with Anomaly Alerts")

# st.markdown(
#     """
#     This intelligent forecasting dashboard predicts grocery demand using
#     ensemble machine learning models and detects abnormal sales behavior
#     using anomaly detection techniques.
#     """
# )


# # =========================
# # Sidebar Filters
# # =========================

# st.sidebar.header("Dashboard Filters")


# product_list = sorted(
#     prediction_df["product_id"].unique()
# )

# selected_product = st.sidebar.selectbox(
#     "Select Product",
#     product_list
# )


# store_list = sorted(
#     prediction_df["store_id"].unique()
# )

# selected_store = st.sidebar.selectbox(
#     "Select Store",
#     store_list
# )


# region_list = sorted(
#     prediction_df["region"].unique()
# )

# selected_region = st.sidebar.selectbox(
#     "Select Region",
#     region_list
# )


# # =========================
# # Filter Dataset
# # =========================

# filtered_df = prediction_df[
#     (
#         prediction_df["product_id"] == selected_product
#     )
#     &
#     (
#         prediction_df["store_id"] == selected_store
#     )
#     &
#     (
#         prediction_df["region"] == selected_region
#     )
# ]


# filtered_df = filtered_df.reset_index(drop=True)


# # =========================
# # Metrics
# # =========================

# mae = mean_absolute_error(
#     filtered_df["demand"],
#     filtered_df["ensemble_prediction"]
# )

# rmse = np.sqrt(
#     mean_squared_error(
#         filtered_df["demand"],
#         filtered_df["ensemble_prediction"]
#     )
# )

# r2 = r2_score(
#     filtered_df["demand"],
#     filtered_df["ensemble_prediction"]
# )

# mape = (
#     np.mean(
#         np.abs(
#             (
#                 filtered_df["demand"] -
#                 filtered_df["ensemble_prediction"]
#             )
#             /
#             filtered_df["demand"]
#         )
#     )
#     * 100
# )

# accuracy = 100 - mape


# st.subheader("Forecast Performance Metrics")

# col1, col2, col3, col4 = st.columns(4)


# col1.metric(
#     "Forecast Accuracy",
#     f"{accuracy:.2f}%"
# )


# col2.metric(
#     "RMSE",
#     f"{rmse:.2f}"
# )


# col3.metric(
#     "MAE",
#     f"{mae:.2f}"
# )


# col4.metric(
#     "R² Score",
#     f"{r2:.4f}"
# )


# # =========================
# # Confidence Band
# # =========================

# residual_std = filtered_df["residual"].std()

# filtered_df["upper_band"] = (
#     filtered_df["ensemble_prediction"] + residual_std
# )

# filtered_df["lower_band"] = (
#     filtered_df["ensemble_prediction"] - residual_std
# )


# # =========================
# # Forecast Plot
# # =========================

# st.subheader("Demand Forecasting Visualization")

# forecast_fig = go.Figure()


# forecast_fig.add_trace(
#     go.Scatter(
#         y=filtered_df["demand"],
#         mode="lines",
#         name="Actual Demand"
#     )
# )


# forecast_fig.add_trace(
#     go.Scatter(
#         y=filtered_df["ensemble_prediction"],
#         mode="lines",
#         name="Forecast"
#     )
# )


# forecast_fig.add_trace(
#     go.Scatter(
#         y=filtered_df["upper_band"],
#         mode="lines",
#         line=dict(width=0),
#         showlegend=False
#     )
# )


# forecast_fig.add_trace(
#     go.Scatter(
#         y=filtered_df["lower_band"],
#         mode="lines",
#         fill="tonexty",
#         line=dict(width=0),
#         name="Confidence Band"
#     )
# )


# anomaly_df = filtered_df[
#     filtered_df["final_anomaly"] == 1
# ]


# forecast_fig.add_trace(
#     go.Scatter(
#         x=anomaly_df.index,
#         y=anomaly_df["demand"],
#         mode="markers",
#         marker=dict(
#             size=10,
#             color="red"
#         ),
#         name="Anomaly Alert"
#     )
# )


# forecast_fig.update_layout(
#     title="Actual vs Forecasted Demand",
#     height=600,
#     hovermode="x unified"
# )


# st.plotly_chart(
#     forecast_fig,
#     use_container_width=True
# )


# # =========================
# # Residual Analysis
# # =========================

# st.subheader("Residual Analysis")

# residual_fig = go.Figure()

# residual_fig.add_trace(
#     go.Scatter(
#         y=filtered_df["residual"],
#         mode="lines",
#         name="Residual"
#     )
# )


# residual_fig.update_layout(
#     title="Residual Distribution Over Time",
#     height=450
# )


# st.plotly_chart(
#     residual_fig,
#     use_container_width=True
# )


# # =========================
# # Demand Distribution
# # =========================

# st.subheader("Demand Distribution")

# hist_fig = px.histogram(
#     filtered_df,
#     x="demand",
#     nbins=30,
#     title="Demand Frequency Distribution"
# )


# st.plotly_chart(
#     hist_fig,
#     use_container_width=True
# )


# # =========================
# # Anomaly Table
# # =========================

# st.subheader("Detected Anomalies")


# if len(anomaly_df) > 0:

#     st.dataframe(
#         anomaly_df[
#             [
#                 "date",
#                 "demand",
#                 "ensemble_prediction",
#                 "residual",
#                 "z_score"
#             ]
#         ],
#         use_container_width=True
#     )

# else:

#     st.success(
#         "No anomalies detected for current filters."
#     )


# # =========================
# # Download Section
# # =========================

# st.subheader("Download Forecast Data")

# csv = filtered_df.to_csv(index=False)

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="forecast_results.csv",
#     mime="text/csv"
# )


# # =========================
# # Footer
# # =========================

# st.markdown("---")

# st.markdown(
#     """
#     ### Machine Learning Models Used

#     - CatBoost Regressor
#     - LightGBM Regressor
#     - Dynamic Weighted Ensemble
#     - LOF Anomaly Detection
#     - Z-Score Validation

#     ### Technologies

#     - Streamlit
#     - Plotly
#     - Scikit-learn
#     - CatBoost
#     - LightGBM
#     """
# )










































# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import joblib
# from datetime import datetime

# # =====================================
# # PAGE CONFIG
# # =====================================

# st.set_page_config(
#     page_title="Grocery Demand Forecasting",
#     page_icon="🛒",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # =====================================
# # CUSTOM CSS
# # =====================================

# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #0f172a;
#     }

#     .stMetric {
#         background-color: #1e293b;
#         padding: 15px;
#         border-radius: 12px;
#         border: 1px solid #334155;
#     }

#     h1, h2, h3 {
#         color: white;
#     }

#     .block-container {
#         padding-top: 2rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # =====================================
# # LOAD MODELS
# # =====================================

# @st.cache_resource

# def load_models():

#     cat_model = joblib.load(
#         "models/catboost_model.pkl"
#     )

#     lgb_model = joblib.load(
#         "models/lightgbm_model.pkl"
#     )

#     label_encoders = joblib.load(
#         "models/label_encoders.pkl"
#     )

#     return cat_model, lgb_model, label_encoders


# cat_model, lgb_model, label_encoders = load_models()


# # =====================================
# # LOAD DATA
# # =====================================

# @st.cache_data

# def load_dataset():

#     df = pd.read_csv(
#         "dataset/sales_data.csv"
#     )

#     return df


# df = load_dataset()


# # =====================================
# # HEADER
# # =====================================

# st.title("🛒 Grocery Demand Forecasting with Anomaly Alerts")

# st.markdown(
#     """
#     Predict grocery demand using an ensemble of CatBoost and LightGBM models.
#     The system also detects unusual demand behavior using anomaly detection logic.
#     """
# )


# # =====================================
# # SIDEBAR
# # =====================================

# st.sidebar.header("⚙️ Prediction Inputs")


# # PRODUCT
# product_list = sorted(df["Product ID"].unique())

# selected_product = st.sidebar.selectbox(
#     "Select Product",
#     product_list
# )


# # STORE
# store_list = sorted(df["Store ID"].unique())

# selected_store = st.sidebar.selectbox(
#     "Select Store",
#     store_list
# )


# # REGION
# region_list = sorted(df["Region"].unique())

# selected_region = st.sidebar.selectbox(
#     "Select Region",
#     region_list
# )


# # CATEGORY
# category_list = sorted(df["Category"].unique())

# selected_category = st.sidebar.selectbox(
#     "Select Category",
#     category_list
# )


# # WEATHER
# weather_list = sorted(df["Weather Condition"].unique())

# selected_weather = st.sidebar.selectbox(
#     "Weather Condition",
#     weather_list
# )


# # SEASONALITY
# season_list = sorted(df["Seasonality"].unique())

# selected_season = st.sidebar.selectbox(
#     "Seasonality",
#     season_list
# )


# # NUMERIC INPUTS
# inventory_level = st.sidebar.slider(
#     "Inventory Level",
#     0,
#     1000,
#     300
# )

# units_sold = st.sidebar.slider(
#     "Units Sold",
#     0,
#     500,
#     120
# )

# units_ordered = st.sidebar.slider(
#     "Units Ordered",
#     0,
#     500,
#     150
# )

# price = st.sidebar.slider(
#     "Price",
#     1.0,
#     1000.0,
#     150.0
# )

# competitor_price = st.sidebar.slider(
#     "Competitor Pricing",
#     1.0,
#     1000.0,
#     145.0
# )


# discount = st.sidebar.slider(
#     "Discount (%)",
#     0,
#     100,
#     10
# )

# promotion = st.sidebar.selectbox(
#     "Promotion",
#     [0, 1]
# )


# epidemic = st.sidebar.selectbox(
#     "Epidemic",
#     [0, 1]
# )


# selected_date = st.sidebar.date_input(
#     "Select Date",
#     datetime.today()
# )


# # =====================================
# # FEATURE ENGINEERING
# # =====================================

# input_df = pd.DataFrame({
#     "Date": [str(selected_date)],
#     "Store ID": [selected_store],
#     "Product ID": [selected_product],
#     "Category": [selected_category],
#     "Region": [selected_region],
#     "Inventory Level": [inventory_level],
#     "Units Sold": [units_sold],
#     "Units Ordered": [units_ordered],
#     "Price": [price],
#     "Discount": [discount],
#     "Weather Condition": [selected_weather],
#     "Promotion": [promotion],
#     "Competitor Pricing": [competitor_price],
#     "Seasonality": [selected_season],
#     "Epidemic": [epidemic]
# })


# # DATE FEATURES
# input_df["Date"] = pd.to_datetime(input_df["Date"])

# input_df["day"] = input_df["Date"].dt.day
# input_df["month"] = input_df["Date"].dt.month
# input_df["dayofweek"] = input_df["Date"].dt.dayofweek
# input_df["week"] = input_df["Date"].dt.isocalendar().week.astype(int)


# # =====================================
# # ENCODING
# # =====================================

# categorical_cols = [
#     "Store ID",
#     "Product ID",
#     "Category",
#     "Region",
#     "Weather Condition",
#     "Seasonality"
# ]


# for col in categorical_cols:

#     if col in label_encoders:

#         input_df[col] = label_encoders[col].transform(
#             input_df[col]
#         )


# # REMOVE DATE
# input_df = input_df.drop(columns=["Date"])


# # =====================================
# # PREDICTION BUTTON
# # =====================================

# predict_button = st.sidebar.button(
#     "🚀 Predict Demand"
# )


# if predict_button:

#     # =====================================
#     # MODEL PREDICTIONS
#     # =====================================

#     cat_pred = cat_model.predict(input_df)[0]

#     lgb_pred = lgb_model.predict(input_df)[0]


#     # ENSEMBLE
#     final_prediction = (
#         0.6 * cat_pred +
#         0.4 * lgb_pred
#     )


#     # =====================================
#     # ANOMALY DETECTION
#     # =====================================

#     historical_mean = df["Demand"].mean()

#     historical_std = df["Demand"].std()

#     z_score = (
#         final_prediction - historical_mean
#     ) / historical_std


#     anomaly_flag = abs(z_score) > 2


#     # =====================================
#     # RESULT SECTION
#     # =====================================

#     st.markdown("---")

#     st.subheader("📈 Forecast Results")


#     col1, col2, col3 = st.columns(3)


#     col1.metric(
#         "CatBoost Prediction",
#         f"{cat_pred:.2f}"
#     )

#     col2.metric(
#         "LightGBM Prediction",
#         f"{lgb_pred:.2f}"
#     )

#     col3.metric(
#         "Ensemble Forecast",
#         f"{final_prediction:.2f}"
#     )


#     # =====================================
#     # ANOMALY ALERT
#     # =====================================

#     st.subheader("🚨 Anomaly Detection")

#     if anomaly_flag:

#         st.error(
#             f"Anomaly Detected! Z-Score = {z_score:.2f}"
#         )

#     else:

#         st.success(
#             f"Normal Demand Pattern | Z-Score = {z_score:.2f}"
#         )


#     # =====================================
#     # GAUGE CHART
#     # =====================================

#     st.subheader("📊 Demand Forecast Gauge")

#     gauge_fig = go.Figure(
#         go.Indicator(
#             mode="gauge+number",
#             value=final_prediction,
#             title={"text": "Predicted Demand"},
#             gauge={
#                 "axis": {"range": [0, 1000]},
#                 "bar": {"color": "cyan"},
#                 "steps": [
#                     {"range": [0, 300], "color": "#14532d"},
#                     {"range": [300, 700], "color": "#78350f"},
#                     {"range": [700, 1000], "color": "#7f1d1d"}
#                 ]
#             }
#         )
#     )

#     gauge_fig.update_layout(
#         height=400
#     )

#     st.plotly_chart(
#         gauge_fig,
#         use_container_width=True
#     )


#     # =====================================
#     # FEATURE SUMMARY
#     # =====================================

#     st.subheader("🧾 Selected Input Summary")

#     display_df = pd.DataFrame({
#         "Feature": [
#             "Product",
#             "Store",
#             "Region",
#             "Category",
#             "Weather",
#             "Seasonality",
#             "Price",
#             "Discount",
#             "Inventory"
#         ],
#         "Value": [
#             selected_product,
#             selected_store,
#             selected_region,
#             selected_category,
#             selected_weather,
#             selected_season,
#             price,
#             discount,
#             inventory_level
#         ]
#     })

#     st.dataframe(
#         display_df,
#         use_container_width=True
#     )


#     # =====================================
#     # HISTORICAL TREND
#     # =====================================

#     st.subheader("📉 Historical Demand Trend")

#     product_history = df[
#         df["Product ID"] == selected_product
#     ]

#     trend_fig = px.line(
#         product_history,
#         x="Date",
#         y="Demand",
#         title="Historical Product Demand"
#     )

#     trend_fig.add_hline(
#         y=final_prediction,
#         line_dash="dash",
#         annotation_text="Forecast"
#     )

#     trend_fig.update_layout(
#         height=500
#     )

#     st.plotly_chart(
#         trend_fig,
#         use_container_width=True
#     )


#     # =====================================
#     # DOWNLOAD RESULT
#     # =====================================

#     result_df = pd.DataFrame({
#         "Predicted Demand": [final_prediction],
#         "CatBoost": [cat_pred],
#         "LightGBM": [lgb_pred],
#         "Z-Score": [z_score],
#         "Anomaly": [anomaly_flag]
#     })

#     csv = result_df.to_csv(index=False)

#     st.download_button(
#         label="⬇ Download Prediction Result",
#         data=csv,
#         file_name="forecast_result.csv",
#         mime="text/csv"
#     )


# # =====================================
# # FOOTER
# # =====================================

# st.markdown("---")

# st.markdown(
#     """
#     ### 🤖 Machine Learning Pipeline

#     - CatBoost Regressor
#     - LightGBM Regressor
#     - Weighted Ensemble Forecasting
#     - Z-Score Based Anomaly Detection

#     ### 🛠 Technologies Used

#     - Streamlit
#     - Plotly
#     - Pandas
#     - Scikit-learn
#     - CatBoost
#     - LightGBM
#     """
# )





















































import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Grocery Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():
    cat_model = joblib.load("models/catboost_model.pkl")
    lgb_model = joblib.load("models/lightgbm_model.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    return cat_model, lgb_model, label_encoders

cat_model, lgb_model, label_encoders = load_models()

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset/sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_dataset()

# =====================================
# TITLE
# =====================================

st.title("🛒 Grocery Demand Forecasting with Anomaly Alerts")

st.markdown("CatBoost + LightGBM Ensemble Based Live Demand Prediction System")

# =====================================
# SIDEBAR INPUTS
# =====================================

st.sidebar.header("⚙️ Input Features")

selected_product = st.sidebar.selectbox("Product ID", sorted(df["Product ID"].unique()))
selected_store = st.sidebar.selectbox("Store ID", sorted(df["Store ID"].unique()))
selected_region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
selected_category = st.sidebar.selectbox("Category", sorted(df["Category"].unique()))
selected_weather = st.sidebar.selectbox("Weather Condition", sorted(df["Weather Condition"].unique()))
selected_season = st.sidebar.selectbox("Seasonality", sorted(df["Seasonality"].unique()))

inventory_level = st.sidebar.slider("Inventory Level", 0, 1000, 300)
units_sold = st.sidebar.slider("Units Sold", 0, 500, 120)
units_ordered = st.sidebar.slider("Units Ordered", 0, 500, 150)
price = st.sidebar.slider("Price", 1.0, 1000.0, 150.0)
competitor_price = st.sidebar.slider("Competitor Price", 1.0, 1000.0, 145.0)
discount = st.sidebar.slider("Discount (%)", 0, 100, 10)
promotion = st.sidebar.selectbox("Promotion", [0, 1])
epidemic = st.sidebar.selectbox("Epidemic", [0, 1])

selected_date = st.sidebar.date_input("Select Date", datetime.today())

# =====================================
# INPUT DATAFRAME
# =====================================

input_df = pd.DataFrame({
    "Product ID": [selected_product],
    "Store ID": [selected_store],
    "Region": [selected_region],
    "Category": [selected_category],
    "Weather Condition": [selected_weather],
    "Seasonality": [selected_season],
    "Inventory Level": [inventory_level],
    "Units Sold": [units_sold],
    "Units Ordered": [units_ordered],
    "Price": [price],
    "Discount": [discount],
    "Promotion": [promotion],
    "Competitor Pricing": [competitor_price],
    "Epidemic": [epidemic]
})

# =====================================
# DATE FEATURES
# =====================================

input_df["Date"] = pd.to_datetime([selected_date])
input_df["day"] = input_df["Date"].dt.day
input_df["month"] = input_df["Date"].dt.month
input_df["dayofweek"] = input_df["Date"].dt.dayofweek
input_df["week"] = input_df["Date"].dt.isocalendar().week.astype(int)

input_df.drop(columns=["Date"], inplace=True)

# =====================================
# SAFE LABEL ENCODING (IMPORTANT FIX)
# =====================================

categorical_cols = [
    "Product ID",
    "Store ID",
    "Region",
    "Category",
    "Weather Condition",
    "Seasonality"
]

for col in categorical_cols:
    if col in label_encoders:
        le = label_encoders[col]

        def safe_encode(x):
            if x in le.classes_:
                return le.transform([x])[0]
            else:
                return 0   # fallback for unseen values

        input_df[col] = input_df[col].apply(safe_encode)

# =====================================
# PREDICTION
# =====================================

if st.sidebar.button("🚀 Predict Demand"):

    cat_pred = cat_model.predict(input_df)[0]
    lgb_pred = lgb_model.predict(input_df)[0]

    final_pred = 0.6 * cat_pred + 0.4 * lgb_pred

    # =====================================
    # ANOMALY DETECTION
    # =====================================

    mean = df["Demand"].mean()
    std = df["Demand"].std()

    z_score = (final_pred - mean) / std

    anomaly = abs(z_score) > 2

    # =====================================
    # RESULT
    # =====================================

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("CatBoost", round(cat_pred, 2))
    col2.metric("LightGBM", round(lgb_pred, 2))
    col3.metric("Final Forecast", round(final_pred, 2))

    if anomaly:
        st.error(f"🚨 Anomaly Detected | Z = {z_score:.2f}")
    else:
        st.success(f"Normal Demand Pattern | Z = {z_score:.2f}")

    # =====================================
    # GAUGE
    # =====================================

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_pred,
        title={"text": "Predicted Demand"},
        gauge={"axis": {"range": [0, 1000]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =====================================
    # TREND PLOT
    # =====================================

    st.subheader("📉 Historical Trend")

    product_history = df[df["Product ID"] == selected_product]

    trend_fig = px.line(
        product_history,
        x="Date",
        y="Demand",
        title="Historical Demand"
    )

    trend_fig.add_hline(
        y=final_pred,
        line_dash="dash",
        annotation_text="Forecast"
    )

    st.plotly_chart(trend_fig, use_container_width=True)

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown("""
### 🧠 ML Pipeline
- CatBoost Regressor  
- LightGBM Regressor  
- Weighted Ensemble  
- Z-Score Anomaly Detection  

### 🛠 Tools
Streamlit | Plotly | Scikit-learn | CatBoost | LightGBM
""")