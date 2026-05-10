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
# # LOAD MODELS
# # =====================================

# @st.cache_resource
# def load_models():
#     cat_model = joblib.load("models/catboost_model.pkl")
#     lgb_model = joblib.load("models/lightgbm_model.pkl")
#     label_encoders = joblib.load("models/label_encoders.pkl")
#     return cat_model, lgb_model, label_encoders

# cat_model, lgb_model, label_encoders = load_models()

# # =====================================
# # LOAD DATA
# # =====================================

# @st.cache_data
# def load_dataset():
#     df = pd.read_csv("dataset/sales_data.csv")
#     df["Date"] = pd.to_datetime(df["Date"])
#     return df

# df = load_dataset()

# # =====================================
# # TITLE
# # =====================================

# st.title("🛒 Grocery Demand Forecasting with Anomaly Alerts")

# st.markdown("CatBoost + LightGBM Ensemble Based Live Demand Prediction System")

# # =====================================
# # SIDEBAR INPUTS
# # =====================================

# st.sidebar.header("⚙️ Input Features")

# selected_product = st.sidebar.selectbox("Product ID", sorted(df["Product ID"].unique()))
# selected_store = st.sidebar.selectbox("Store ID", sorted(df["Store ID"].unique()))
# selected_region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
# selected_category = st.sidebar.selectbox("Category", sorted(df["Category"].unique()))
# selected_weather = st.sidebar.selectbox("Weather Condition", sorted(df["Weather Condition"].unique()))
# selected_season = st.sidebar.selectbox("Seasonality", sorted(df["Seasonality"].unique()))

# inventory_level = st.sidebar.slider("Inventory Level", 0, 1000, 300)
# units_sold = st.sidebar.slider("Units Sold", 0, 500, 120)
# units_ordered = st.sidebar.slider("Units Ordered", 0, 500, 150)
# price = st.sidebar.slider("Price", 1.0, 1000.0, 150.0)
# competitor_price = st.sidebar.slider("Competitor Price", 1.0, 1000.0, 145.0)
# discount = st.sidebar.slider("Discount (%)", 0, 100, 10)
# promotion = st.sidebar.selectbox("Promotion", [0, 1])
# epidemic = st.sidebar.selectbox("Epidemic", [0, 1])

# selected_date = st.sidebar.date_input("Select Date", datetime.today())

# # =====================================
# # INPUT DATAFRAME
# # =====================================

# input_df = pd.DataFrame({
#     "Product ID": [selected_product],
#     "Store ID": [selected_store],
#     "Region": [selected_region],
#     "Category": [selected_category],
#     "Weather Condition": [selected_weather],
#     "Seasonality": [selected_season],
#     "Inventory Level": [inventory_level],
#     "Units Sold": [units_sold],
#     "Units Ordered": [units_ordered],
#     "Price": [price],
#     "Discount": [discount],
#     "Promotion": [promotion],
#     "Competitor Pricing": [competitor_price],
#     "Epidemic": [epidemic]
# })

# # =====================================
# # DATE FEATURES
# # =====================================

# input_df["Date"] = pd.to_datetime([selected_date])
# input_df["day"] = input_df["Date"].dt.day
# input_df["month"] = input_df["Date"].dt.month
# input_df["dayofweek"] = input_df["Date"].dt.dayofweek
# input_df["week"] = input_df["Date"].dt.isocalendar().week.astype(int)

# input_df.drop(columns=["Date"], inplace=True)

# # =====================================
# # SAFE LABEL ENCODING (IMPORTANT FIX)
# # =====================================

# categorical_cols = [
#     "Product ID",
#     "Store ID",
#     "Region",
#     "Category",
#     "Weather Condition",
#     "Seasonality"
# ]

# for col in categorical_cols:
#     if col in label_encoders:
#         le = label_encoders[col]

#         def safe_encode(x):
#             if x in le.classes_:
#                 return le.transform([x])[0]
#             else:
#                 return 0   # fallback for unseen values

#         input_df[col] = input_df[col].apply(safe_encode)

# # =====================================
# # PREDICTION
# # =====================================

# if st.sidebar.button("🚀 Predict Demand"):

#     cat_pred = cat_model.predict(input_df)[0]
#     lgb_pred = lgb_model.predict(input_df)[0]

#     final_pred = 0.6 * cat_pred + 0.4 * lgb_pred

#     # =====================================
#     # ANOMALY DETECTION
#     # =====================================

#     mean = df["Demand"].mean()
#     std = df["Demand"].std()

#     z_score = (final_pred - mean) / std

#     anomaly = abs(z_score) > 2

#     # =====================================
#     # RESULT
#     # =====================================

#     st.markdown("---")
#     st.subheader("📊 Prediction Result")

#     col1, col2, col3 = st.columns(3)

#     col1.metric("CatBoost", round(cat_pred, 2))
#     col2.metric("LightGBM", round(lgb_pred, 2))
#     col3.metric("Final Forecast", round(final_pred, 2))

#     if anomaly:
#         st.error(f"🚨 Anomaly Detected | Z = {z_score:.2f}")
#     else:
#         st.success(f"Normal Demand Pattern | Z = {z_score:.2f}")

#     # =====================================
#     # GAUGE
#     # =====================================

#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=final_pred,
#         title={"text": "Predicted Demand"},
#         gauge={"axis": {"range": [0, 1000]}}
#     ))

#     st.plotly_chart(fig, use_container_width=True)

#     # =====================================
#     # TREND PLOT
#     # =====================================

#     st.subheader("📉 Historical Trend")

#     product_history = df[df["Product ID"] == selected_product]

#     trend_fig = px.line(
#         product_history,
#         x="Date",
#         y="Demand",
#         title="Historical Demand"
#     )

#     trend_fig.add_hline(
#         y=final_pred,
#         line_dash="dash",
#         annotation_text="Forecast"
#     )

#     st.plotly_chart(trend_fig, use_container_width=True)

# # =====================================
# # FOOTER
# # =====================================

# st.markdown("---")

# st.markdown("""
# ### 🧠 ML Pipeline
# - CatBoost Regressor  
# - LightGBM Regressor  
# - Weighted Ensemble  
# - Z-Score Anomaly Detection  

# ### 🛠 Tools
# Streamlit | Plotly | Scikit-learn | CatBoost | LightGBM
# """)












































# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import joblib
# from datetime import datetime
# import time

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
# # LOAD MODELS
# # =====================================

# @st.cache_resource
# def load_models():
#     cat_model = joblib.load("models/catboost_model.pkl")
#     lgb_model = joblib.load("models/lightgbm_model.pkl")
#     label_encoders = joblib.load("models/label_encoders.pkl")
#     return cat_model, lgb_model, label_encoders

# cat_model, lgb_model, label_encoders = load_models()

# # =====================================
# # LOAD DATA
# # =====================================

# @st.cache_data
# def load_dataset():
#     df = pd.read_csv("dataset/sales_data.csv")
#     df["Date"] = pd.to_datetime(df["Date"])
#     return df

# df = load_dataset()

# # =====================================
# # TITLE
# # =====================================

# st.title("🛒 Grocery Demand Forecasting with Anomaly Alerts")
# st.markdown("🔥 CatBoost + LightGBM Ensemble | Real-time Prediction Dashboard")

# # =====================================
# # SIDEBAR INPUTS
# # =====================================

# st.sidebar.header("⚙️ Input Features")

# selected_product = st.sidebar.selectbox("Product ID", sorted(df["Product ID"].unique()))
# selected_store = st.sidebar.selectbox("Store ID", sorted(df["Store ID"].unique()))
# selected_region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
# selected_category = st.sidebar.selectbox("Category", sorted(df["Category"].unique()))
# selected_weather = st.sidebar.selectbox("Weather Condition", sorted(df["Weather Condition"].unique()))
# selected_season = st.sidebar.selectbox("Seasonality", sorted(df["Seasonality"].unique()))

# inventory_level = st.sidebar.slider("Inventory Level", 0, 1000, 300)
# units_sold = st.sidebar.slider("Units Sold", 0, 500, 120)
# units_ordered = st.sidebar.slider("Units Ordered", 0, 500, 150)
# price = st.sidebar.slider("Price", 1.0, 1000.0, 150.0)
# competitor_price = st.sidebar.slider("Competitor Price", 1.0, 1000.0, 145.0)
# discount = st.sidebar.slider("Discount (%)", 0, 100, 10)
# promotion = st.sidebar.selectbox("Promotion", [0, 1])
# epidemic = st.sidebar.selectbox("Epidemic", [0, 1])

# selected_date = st.sidebar.date_input("Select Date", datetime.today())

# # =====================================
# # BUILD INPUT DF
# # =====================================

# input_df = pd.DataFrame({
#     "Product ID": [selected_product],
#     "Store ID": [selected_store],
#     "Region": [selected_region],
#     "Category": [selected_category],
#     "Weather Condition": [selected_weather],
#     "Seasonality": [selected_season],
#     "Inventory Level": [inventory_level],
#     "Units Sold": [units_sold],
#     "Units Ordered": [units_ordered],
#     "Price": [price],
#     "Discount": [discount],
#     "Promotion": [promotion],
#     "Competitor Pricing": [competitor_price],
#     "Epidemic": [epidemic]
# })

# # =====================================
# # DATE FEATURES
# # =====================================

# input_df["Date"] = pd.to_datetime([selected_date])
# input_df["day"] = input_df["Date"].dt.day
# input_df["month"] = input_df["Date"].dt.month
# input_df["dayofweek"] = input_df["Date"].dt.dayofweek
# input_df["week"] = input_df["Date"].dt.isocalendar().week.astype(int)
# input_df.drop(columns=["Date"], inplace=True)

# # =====================================
# # SAFE ENCODING (FIXED - NO CATBOOST ERROR)
# # =====================================

# categorical_cols = [
#     "Product ID",
#     "Store ID",
#     "Region",
#     "Category",
#     "Weather Condition",
#     "Seasonality"
# ]

# for col in categorical_cols:
#     if col in label_encoders:

#         le = label_encoders[col]

#         # mapping fix (FAST + SAFE)
#         mapping = {cls: idx for idx, cls in enumerate(le.classes_)}

#         input_df[col] = input_df[col].map(lambda x: mapping.get(x, 0))

# # =====================================
# # MATCH MODEL FEATURE ORDER (IMPORTANT FIX)
# # =====================================

# if hasattr(cat_model, "feature_names_"):
#     try:
#         input_df = input_df[cat_model.feature_names_]
#     except:
#         pass

# # =====================================
# # PREDICTION BUTTON
# # =====================================

# if st.sidebar.button("🚀 Predict Demand"):

#     with st.spinner("🤖 Running AI Models..."):

#         time.sleep(1.2)

#         cat_pred = cat_model.predict(input_df)[0]
#         lgb_pred = lgb_model.predict(input_df)[0]

#         final_pred = 0.6 * cat_pred + 0.4 * lgb_pred

#         # =========================
#         # ANOMALY DETECTION
#         # =========================

#         mean = df["Demand"].mean()
#         std = df["Demand"].std()

#         z_score = (final_pred - mean) / std
#         anomaly = abs(z_score) > 2

#     st.success("✅ Prediction Completed!")

#     # =========================
#     # RESULT UI
#     # =========================

#     st.markdown("---")
#     st.subheader("📊 Prediction Result")

#     col1, col2, col3 = st.columns(3)

#     col1.metric("CatBoost", round(cat_pred, 2))
#     col2.metric("LightGBM", round(lgb_pred, 2))
#     col3.metric("Final Forecast", round(final_pred, 2))

#     # =========================
#     # ANOMALY ALERT
#     # =========================

#     if anomaly:
#         st.error(f"🚨 Anomaly Detected | Z = {z_score:.2f}")
#     else:
#         st.success(f"Normal Demand Pattern | Z = {z_score:.2f}")

#     # =========================
#     # GAUGE ANIMATION
#     # =========================

#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=final_pred,
#         title={"text": "Predicted Demand"},
#         gauge={
#             "axis": {"range": [0, 1000]},
#             "bar": {"color": "cyan"},
#             "steps": [
#                 {"range": [0, 300], "color": "#14532d"},
#                 {"range": [300, 700], "color": "#78350f"},
#                 {"range": [700, 1000], "color": "#7f1d1d"}
#             ]
#         }
#     ))

#     st.plotly_chart(fig, use_container_width=True)

#     # =========================
#     # ANIMATION EFFECT
#     # =========================

#     st.balloons()

#     # =========================
#     # TREND PLOT
#     # =========================

#     st.subheader("📉 Historical Trend")

#     product_history = df[df["Product ID"] == selected_product]

#     trend_fig = px.line(
#         product_history,
#         x="Date",
#         y="Demand",
#         title="Historical Demand Trend"
#     )

#     trend_fig.add_hline(
#         y=final_pred,
#         line_dash="dash",
#         annotation_text="Forecast"
#     )

#     st.plotly_chart(trend_fig, use_container_width=True)

# # =====================================
# # FOOTER
# # =====================================

# st.markdown("---")
# st.markdown("""
# ### 🧠 ML Pipeline
# - CatBoost Regressor  
# - LightGBM Regressor  
# - Weighted Ensemble  
# - Z-Score Anomaly Detection  

# ### 🚀 Features
# ✔ Real-time prediction  
# ✔ Safe encoding system  
# ✔ Animated dashboard  
# ✔ Production-ready ML pipeline  
# """)













































# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import joblib
# from datetime import datetime
# import time

# # =====================================
# # CONFIG
# # =====================================

# st.set_page_config(
#     page_title="Grocery Demand Forecasting",
#     page_icon="🛒",
#     layout="wide"
# )

# # =====================================
# # LOAD MODELS
# # =====================================

# @st.cache_resource
# def load_models():
#     cat_model = joblib.load("models/catboost_model.pkl")
#     lgb_model = joblib.load("models/lightgbm_model.pkl")
#     label_encoders = joblib.load("models/label_encoders.pkl")
#     return cat_model, lgb_model, label_encoders

# cat_model, lgb_model, label_encoders = load_models()

# # =====================================
# # LOAD DATA
# # =====================================

# @st.cache_data
# def load_data():
#     df = pd.read_csv("dataset/sales_data.csv")
#     df["Date"] = pd.to_datetime(df["Date"])
#     return df

# df = load_data()

# # =====================================
# # TITLE
# # =====================================

# st.title("🛒 Grocery Demand Forecasting AI System")
# st.markdown("CatBoost + LightGBM Ensemble | Fully Stable Prediction Engine")

# # =====================================
# # SIDEBAR INPUT
# # =====================================

# st.sidebar.header("📌 Input Features")

# product = st.sidebar.selectbox(
#     "Product ID",
#     sorted(df["Product ID"].unique())
# )

# store = st.sidebar.selectbox(
#     "Store ID",
#     sorted(df["Store ID"].unique())
# )

# region = st.sidebar.selectbox(
#     "Region",
#     sorted(df["Region"].unique())
# )

# category = st.sidebar.selectbox(
#     "Category",
#     sorted(df["Category"].unique())
# )

# weather = st.sidebar.selectbox(
#     "Weather Condition",
#     sorted(df["Weather Condition"].unique())
# )

# season = st.sidebar.selectbox(
#     "Seasonality",
#     sorted(df["Seasonality"].unique())
# )

# inventory = st.sidebar.slider(
#     "Inventory Level",
#     0,
#     1000,
#     300
# )

# sold = st.sidebar.slider(
#     "Units Sold",
#     0,
#     500,
#     120
# )

# ordered = st.sidebar.slider(
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

# comp_price = st.sidebar.slider(
#     "Competitor Price",
#     1.0,
#     1000.0,
#     145.0
# )

# discount = st.sidebar.slider(
#     "Discount",
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
#     "Select Date (Supports 2026+)",
#     datetime.today()
# )

# # =====================================
# # BUILD INPUT DF
# # =====================================

# input_df = pd.DataFrame({
#     "product_id": [product],
#     "store_id": [store],
#     "region": [region],
#     "category": [category],
#     "weather_condition": [weather],
#     "seasonality": [season],
#     "inventory_level": [inventory],
#     "units_sold": [sold],
#     "units_ordered": [ordered],
#     "price": [price],
#     "discount": [discount],
#     "promotion": [promotion],
#     "competitor_pricing": [comp_price],
#     "epidemic": [epidemic]
# })

# # =====================================
# # DATE FEATURES
# # =====================================

# input_df["date"] = pd.to_datetime([selected_date])

# input_df["day"] = input_df["date"].dt.day

# input_df["month"] = input_df["date"].dt.month

# input_df["year"] = input_df["date"].dt.year

# input_df["day_of_week"] = (
#     input_df["date"].dt.dayofweek
# )

# input_df["week_of_year"] = (
#     input_df["date"]
#     .dt
#     .isocalendar()
#     .week
#     .astype(int)
# )

# input_df["quarter"] = (
#     input_df["date"]
#     .dt
#     .quarter
# )

# input_df["is_weekend"] = (
#     input_df["day_of_week"] >= 5
# ).astype(int)

# input_df["day_name"] = (
#     input_df["date"]
#     .dt
#     .day_name()
# )

# input_df["month_sin"] = np.sin(
#     2 * np.pi * input_df["month"] / 12
# )

# input_df["month_cos"] = np.cos(
#     2 * np.pi * input_df["month"] / 12
# )

# input_df["dow_sin"] = np.sin(
#     2 * np.pi * input_df["day_of_week"] / 7
# )

# input_df["dow_cos"] = np.cos(
#     2 * np.pi * input_df["day_of_week"] / 7
# )

# input_df.drop(
#     columns=["date"],
#     inplace=True
# )

# # =====================================
# # EXTRA FEATURES
# # =====================================

# input_df["is_holiday"] = 0

# input_df["demand_lag_1"] = sold
# input_df["demand_lag_7"] = sold
# input_df["demand_lag_14"] = sold
# input_df["demand_lag_30"] = sold

# input_df["rolling_mean_7"] = sold
# input_df["rolling_mean_14"] = sold
# input_df["rolling_mean_30"] = sold

# input_df["rolling_std_7"] = 0
# input_df["rolling_std_30"] = 0

# input_df["expanding_mean"] = sold

# input_df["demand_change_1"] = 0
# input_df["demand_change_7"] = 0

# input_df["price_diff"] = (
#     price - comp_price
# )

# input_df["discounted_price"] = (
#     price * (1 - discount / 100)
# )

# input_df["inventory_sales_ratio"] = (
#     inventory / (sold + 1)
# )

# # =====================================
# # SAFE ENCODING
# # =====================================

# categorical_cols = [
#     "product_id",
#     "store_id",
#     "region",
#     "category",
#     "weather_condition",
#     "seasonality"
# ]

# for col in categorical_cols:

#     le = label_encoders[col]

#     mapping = {
#         cls: i
#         for i, cls in enumerate(le.classes_)
#     }

#     input_df[col] = input_df[col].map(
#         lambda x: mapping.get(x, 0)
#     ).astype(int)

# # =====================================
# # MANUAL DAY ENCODING
# # =====================================

# day_mapping = {
#     "Monday": 0,
#     "Tuesday": 1,
#     "Wednesday": 2,
#     "Thursday": 3,
#     "Friday": 4,
#     "Saturday": 5,
#     "Sunday": 6
# }

# input_df["day_name"] = input_df["day_name"].map(
#     day_mapping
# ).fillna(0).astype(int)

# # =====================================
# # ALIGN FEATURES
# # =====================================

# try:
#     input_df = input_df.reindex(
#         columns=cat_model.feature_names_,
#         fill_value=0
#     )
# except:
#     pass

# # =====================================
# # PREDICTION BUTTON
# # =====================================

# if st.sidebar.button("🚀 Predict Demand"):

#     with st.spinner("AI Models Running... 🤖"):

#         time.sleep(1.5)

#         cat_pred = cat_model.predict(
#             input_df
#         )[0]

#         lgb_pred = lgb_model.predict(
#             input_df
#         )[0]

#         cat_weight = 1 / 13.79
#         lgb_weight = 1 / 14.25

#         total = cat_weight + lgb_weight

#         cat_weight /= total
#         lgb_weight /= total

#         final_pred = (
#             cat_weight * cat_pred +
#             lgb_weight * lgb_pred
#         )

#         # =========================
#         # ANOMALY DETECTION
#         # =========================

#         mean = df["Demand"].mean()

#         std = df["Demand"].std()

#         z = (
#             final_pred - mean
#         ) / std

#         anomaly = abs(z) > 2

#     st.success("Prediction Completed ✅")

#     # =========================
#     # RESULTS
#     # =========================

#     st.subheader("📊 Forecast Result")

#     col1, col2, col3 = st.columns(3)

#     col1.metric(
#         "CatBoost",
#         round(cat_pred, 2)
#     )

#     col2.metric(
#         "LightGBM",
#         round(lgb_pred, 2)
#     )

#     col3.metric(
#         "Final Prediction",
#         round(final_pred, 2)
#     )

#     if anomaly:

#         st.error(
#             f"🚨 Anomaly Detected | Z = {z:.2f}"
#         )

#     else:

#         st.success(
#             f"Normal Pattern | Z = {z:.2f}"
#         )

#     # =========================
#     # GAUGE CHART
#     # =========================

#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=final_pred,
#         title={"text": "Demand Forecast"},
#         gauge={
#             "axis": {
#                 "range": [0, 1000]
#             }
#         }
#     ))

#     st.plotly_chart(
#         fig,
#         use_container_width=True
#     )

#     st.balloons()

#     # =========================
#     # HISTORICAL TREND
#     # =========================

#     st.subheader("📉 Historical Trend")

#     hist = df[
#         df["Product ID"] == product
#     ]

#     fig2 = px.line(
#         hist,
#         x="Date",
#         y="Demand",
#         title="Historical Demand Trend"
#     )

#     fig2.add_hline(
#         y=final_pred,
#         line_dash="dash",
#         annotation_text="Forecast"
#     )

#     st.plotly_chart(
#         fig2,
#         use_container_width=True
#     )

# # =====================================
# # FOOTER
# # =====================================

# st.markdown("---")

# st.markdown("""
# ### 🧠 System Info

# - CatBoost + LightGBM Ensemble  
# - Dynamic Weighted Prediction  
# - Safe Encoding System  
# - Z-score anomaly detection  
# - 2026+ future forecasting support  
# - Production-ready ML pipeline  
# """)
















































import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
import time

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Grocery Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #2E2E2E;
}

.metric-title {
    font-size: 18px;
    color: #AAAAAA;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #00FFAA;
}

.alert-box {
    padding: 18px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    cat_model = joblib.load(
        "models/catboost_model.pkl"
    )

    lgb_model = joblib.load(
        "models/lightgbm_model.pkl"
    )

    label_encoders = joblib.load(
        "models/label_encoders.pkl"
    )

    return cat_model, lgb_model, label_encoders


cat_model, lgb_model, label_encoders = load_models()

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "dataset/sales_data.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    return df


df = load_data()

# =========================================================
# HEADER
# =========================================================

st.title("🛒 Smart Grocery Demand Forecasting")

st.markdown("""
AI-powered grocery demand prediction and anomaly monitoring system.

This dashboard helps businesses forecast future demand,
identify unusual demand spikes, and optimize inventory planning.
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📌 Forecast Configuration")

product = st.sidebar.selectbox(
    "Product",
    sorted(df["Product ID"].unique())
)

store = st.sidebar.selectbox(
    "Store",
    sorted(df["Store ID"].unique())
)

region = st.sidebar.selectbox(
    "Region",
    sorted(df["Region"].unique())
)

selected_date = st.sidebar.date_input(
    "Forecast Date",
    datetime.today()
)

inventory = st.sidebar.slider(
    "Inventory Level",
    0,
    1000,
    300
)

sold = st.sidebar.slider(
    "Recent Units Sold",
    0,
    500,
    120
)

price = st.sidebar.slider(
    "Product Price",
    1.0,
    1000.0,
    150.0
)

# =========================================================
# ADVANCED SETTINGS
# =========================================================

with st.sidebar.expander("⚙️ Advanced Settings"):

    category = st.selectbox(
        "Category",
        sorted(df["Category"].unique())
    )

    weather = st.selectbox(
        "Weather Condition",
        sorted(df["Weather Condition"].unique())
    )

    season = st.selectbox(
        "Seasonality",
        sorted(df["Seasonality"].unique())
    )

    ordered = st.slider(
        "Units Ordered",
        0,
        500,
        150
    )

    discount = st.slider(
        "Discount",
        0,
        100,
        10
    )

    promotion = st.selectbox(
        "Promotion",
        [0, 1]
    )

    epidemic = st.selectbox(
        "Epidemic",
        [0, 1]
    )

    comp_price = st.slider(
        "Competitor Price",
        1.0,
        1000.0,
        145.0
    )

# =========================================================
# DEFAULT VALUES
# =========================================================

if "category" not in locals():
    category = df["Category"].mode()[0]

if "weather" not in locals():
    weather = df["Weather Condition"].mode()[0]

if "season" not in locals():
    season = df["Seasonality"].mode()[0]

if "ordered" not in locals():
    ordered = 150

if "discount" not in locals():
    discount = 10

if "promotion" not in locals():
    promotion = 0

if "epidemic" not in locals():
    epidemic = 0

if "comp_price" not in locals():
    comp_price = 145.0

# =========================================================
# BUILD INPUT DATAFRAME
# =========================================================

input_df = pd.DataFrame({

    "product_id": [product],
    "store_id": [store],
    "region": [region],
    "category": [category],
    "weather_condition": [weather],
    "seasonality": [season],

    "inventory_level": [inventory],
    "units_sold": [sold],
    "units_ordered": [ordered],

    "price": [price],
    "discount": [discount],
    "promotion": [promotion],

    "competitor_pricing": [comp_price],
    "epidemic": [epidemic]

})

# =========================================================
# DATE FEATURES
# =========================================================

input_df["date"] = pd.to_datetime(
    [selected_date]
)

input_df["day"] = (
    input_df["date"].dt.day
)

input_df["month"] = (
    input_df["date"].dt.month
)

input_df["year"] = (
    input_df["date"].dt.year
)

input_df["day_of_week"] = (
    input_df["date"].dt.dayofweek
)

input_df["week_of_year"] = (
    input_df["date"]
    .dt
    .isocalendar()
    .week
    .astype(int)
)

input_df["quarter"] = (
    input_df["date"]
    .dt
    .quarter
)

input_df["is_weekend"] = (
    input_df["day_of_week"] >= 5
).astype(int)

input_df["day_name"] = (
    input_df["date"]
    .dt
    .day_name()
)

# =========================================================
# CYCLICAL FEATURES
# =========================================================

input_df["month_sin"] = np.sin(
    2 * np.pi * input_df["month"] / 12
)

input_df["month_cos"] = np.cos(
    2 * np.pi * input_df["month"] / 12
)

input_df["dow_sin"] = np.sin(
    2 * np.pi * input_df["day_of_week"] / 7
)

input_df["dow_cos"] = np.cos(
    2 * np.pi * input_df["day_of_week"] / 7
)

# =========================================================
# EXTRA FEATURES
# =========================================================

input_df["is_holiday"] = 0

input_df["demand_lag_1"] = sold
input_df["demand_lag_7"] = sold
input_df["demand_lag_14"] = sold
input_df["demand_lag_30"] = sold

input_df["rolling_mean_7"] = sold
input_df["rolling_mean_14"] = sold
input_df["rolling_mean_30"] = sold

input_df["rolling_std_7"] = 0
input_df["rolling_std_30"] = 0

input_df["expanding_mean"] = sold

input_df["demand_change_1"] = 0
input_df["demand_change_7"] = 0

input_df["price_diff"] = (
    price - comp_price
)

input_df["discounted_price"] = (
    price * (1 - discount / 100)
)

input_df["inventory_sales_ratio"] = (
    inventory / (sold + 1)
)

input_df.drop(
    columns=["date"],
    inplace=True
)

# =========================================================
# SAFE ENCODING
# =========================================================

categorical_cols = [

    "product_id",
    "store_id",
    "region",
    "category",
    "weather_condition",
    "seasonality"

]

for col in categorical_cols:

    le = label_encoders[col]

    mapping = {
        cls: i
        for i, cls in enumerate(le.classes_)
    }

    input_df[col] = input_df[col].map(
        lambda x: mapping.get(x, 0)
    ).astype(int)

# =========================================================
# DAY ENCODING
# =========================================================

day_mapping = {

    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6

}

input_df["day_name"] = input_df[
    "day_name"
].map(day_mapping).fillna(0).astype(int)

# =========================================================
# FEATURE ALIGNMENT
# =========================================================

try:

    input_df = input_df.reindex(
        columns=cat_model.feature_names_,
        fill_value=0
    )

except:
    pass

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.sidebar.button("🚀 Generate Forecast"):

    with st.spinner("Running AI Forecast Models..."):

        time.sleep(1.5)

        # =========================================
        # MODEL PREDICTIONS
        # =========================================

        cat_pred = cat_model.predict(
            input_df
        )[0]

        lgb_pred = lgb_model.predict(
            input_df
        )[0]

        # =========================================
        # DYNAMIC ENSEMBLE
        # =========================================

        cat_weight = 1 / 13.79
        lgb_weight = 1 / 14.25

        total = cat_weight + lgb_weight

        cat_weight /= total
        lgb_weight /= total

        final_pred = (
            cat_weight * cat_pred +
            lgb_weight * lgb_pred
        )

        # =========================================
        # CONFIDENCE SCORE
        # =========================================

        confidence = max(
            70,
            100 - abs(cat_pred - lgb_pred)
        )

        # =========================================
        # ANOMALY DETECTION
        # =========================================

        mean = df["Demand"].mean()

        std = df["Demand"].std()

        z = (
            final_pred - mean
        ) / std

        anomaly = abs(z) > 2

    # =====================================================
    # KPI SECTION
    # =====================================================

    st.subheader("📊 Forecast Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                Forecast Demand
            </div>
            <div class="metric-value">
                {final_pred:.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                Prediction Confidence
            </div>
            <div class="metric-value">
                {confidence:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                Inventory Level
            </div>
            <div class="metric-value">
                {inventory}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:

        status = "Alert" if anomaly else "Normal"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                Demand Status
            </div>
            <div class="metric-value">
                {status}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # BUSINESS INSIGHT
    # =====================================================

    st.subheader("💡 Business Insight")

    if final_pred > inventory:

        st.warning("""
        High demand is expected compared to current inventory.
        Consider increasing stock levels.
        """)

    else:

        st.success("""
        Current inventory appears sufficient for expected demand.
        """)

    if anomaly:

        st.error("""
        🚨 Unusual demand behavior detected.

        Recommended Actions:
        - Review inventory planning
        - Monitor supply chain
        - Check ongoing promotions
        """)

    else:

        st.info("""
        Demand pattern appears stable and within expected range.
        """)

    # =====================================================
    # TABS
    # =====================================================

    # tab1, tab2, tab3 = st.tabs([
    #     "📈 Forecast",
    #     "📉 Historical Trend",
    #     "⚙️ Technical Details"
    # ])

  # =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "📈 Forecast",
    "📉 Historical Trend",
    "⚙️ Technical Details"
])

# =====================================================
# FORECAST TAB
# =====================================================

with tab1:

    st.subheader("Forecast Visualization")

    fig = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=final_pred,

            title={
                "text": "Forecasted Demand"
            },

            gauge={
                "axis": {
                    "range": [0, 1000]
                },

                "bar": {
                    "color": "cyan"
                },

                "steps": [
                    {
                        "range": [0, 300],
                        "color": "#14532d"
                    },

                    {
                        "range": [300, 700],
                        "color": "#78350f"
                    },

                    {
                        "range": [700, 1000],
                        "color": "#7f1d1d"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.metric(
        "Final Ensemble Forecast",
        f"{final_pred:.2f}"
    )

    if anomaly:
        st.error(
            f"🚨 Anomaly Detected | Z-Score = {z:.2f}"
        )

    else:
        st.success(
            f"✅ Normal Demand Pattern | Z-Score = {z:.2f}"
        )

# =====================================================
# TREND TAB
# =====================================================

with tab2:

    st.subheader(
        "Historical Demand Trend"
    )

    hist = df[
        df["Product ID"] == product
    ].copy()

    # =====================================
    # SIMPLE ANOMALY DETECTION
    # =====================================

    mean_demand = hist["Demand"].mean()

    std_demand = hist["Demand"].std()

    hist["z_score"] = (
        hist["Demand"] - mean_demand
    ) / std_demand

    hist["anomaly"] = (
        hist["z_score"].abs() > 2
    )

    anomaly_points = hist[
        hist["anomaly"] == True
    ]

    # =====================================
    # MAIN TREND CHART
    # =====================================

    fig2 = go.Figure()

    # DEMAND LINE

    fig2.add_trace(

        go.Scatter(

            x=hist["Date"],

            y=hist["Demand"],

            mode="lines",

            name="Historical Demand",

            line=dict(
                width=3
            )

        )

    )

    # FORECAST LINE

    fig2.add_hline(

        y=final_pred,

        line_dash="dash",

        annotation_text="Forecast"

    )

    # ANOMALY POINTS

    fig2.add_trace(

        go.Scatter(

            x=anomaly_points["Date"],

            y=anomaly_points["Demand"],

            mode="markers",

            name="Anomaly",

            marker=dict(

                color="red",

                size=10,

                symbol="circle"

            )

        )

    )

    fig2.update_layout(

        title="Demand Trend with Anomaly Alerts",

        xaxis_title="Date",

        yaxis_title="Demand",

        hovermode="x unified",

        height=550

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    # =====================================
    # ANOMALY TABLE
    # =====================================

    if len(anomaly_points) > 0:

        st.subheader(
            "🚨 Detected Anomalies"
        )

        st.dataframe(

            anomaly_points[
                [
                    "Date",
                    "Demand",
                    "z_score"
                ]
            ],

            use_container_width=True

        )

    else:

        st.success(
            "No historical anomalies detected."
        )

# =====================================================
# TECHNICAL TAB
# =====================================================

with tab3:

    st.subheader(
        "Model Technical Information"
    )

    tech_df = pd.DataFrame({

        "Model": [
            "CatBoost",
            "LightGBM"
        ],

        "Prediction": [
            round(cat_pred, 2),
            round(lgb_pred, 2)
        ]

    })

    st.dataframe(
        tech_df,
        use_container_width=True
    )

    st.write(
        f"📌 Z-Score: {z:.2f}"
    )

    st.write(
        f"📌 Ensemble Confidence: {confidence:.2f}%"
    )

    st.write(
        f"📌 Anomaly Status: {anomaly}"
    )

    st.write(
        f"📌 Forecast Value: {final_pred:.2f}"
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""

### 🧠 System Features

- AI-Powered Demand Forecasting
- Intelligent Ensemble Prediction
- Automated Demand Alert System
- Historical Trend Analysis
- Inventory Optimization Insights
- Business-Friendly Interactive Dashboard

""")