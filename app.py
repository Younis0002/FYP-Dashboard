import streamlit as st
import pandas as pd
import os
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="CryptoCurrency Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 CryptoCurrency Price Prediction Using Deep Learning Models Dashboard")
st.markdown("### Research-Based Comparative Analysis of RNN & Hybrid Architectures")
st.markdown("---")

# --------------------------------------------------
# DARK / LIGHT MODE TOGGLE (Fixed)
# --------------------------------------------------
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    template = "plotly_dark"
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #0e1117;
            color: white;
        }
        .stDataFrame table {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    template = "plotly_white"
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: white;
            color: black;
        }
        .stDataFrame table {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# MODEL METRICS
# --------------------------------------------------
model_metrics = {
    "cmamba": {"RMSE": 1594.38, "MAE": 1142.30, "MAPE": 2.14, "R2": 0.9868},
    "lstm": {"RMSE": 2618.0602, "MAE": 1929.6820, "MAPE": 3.4240, "R2": 0.9645},
    "gru": {"RMSE": 2341.8926, "MAE": 1813.1152, "MAPE": 3.1891, "R2": 0.9716},
    "bilstm": {"RMSE": 1866.7829, "MAE": 1327.9761, "MAPE": 2.4112, "R2": 0.9819},
    "cryptomamba_attngatedmlp_v": {"RMSE": 1718.78, "MAE": 1235.73, "MAPE": 2.25, "R2": 0.9847},
    "cryptomamba_fouriermlp_v": {"RMSE": 1992.26, "MAE": 1455.89, "MAPE": 2.69, "R2": 0.9794},
    "cryptomamba_tcn_v": {"RMSE": 1805.19, "MAE": 1322.50, "MAPE": 2.44, "R2": 0.9800},
    "cryptomamba_attentionmlp_v": {"RMSE": 1491.02, "MAE": 1029.74, "MAPE": 1.90, "R2": 0.9885}
}

# --------------------------------------------------
# PREDICTIONS FOLDER CHECK
# --------------------------------------------------
prediction_folder = "predictions"
if not os.path.exists(prediction_folder):
    st.error("❌ 'predictions' folder not found.")
    st.stop()

files = [f for f in os.listdir(prediction_folder) if f.endswith("_predictions.csv")]
if len(files) == 0:
    st.error("❌ No prediction files found inside 'predictions' folder.")
    st.stop()

# --------------------------------------------------
# MAP FILENAMES TO METRICS KEYS
# --------------------------------------------------
file_to_metrics_key = {
    "CryptoMamba_AttentionMLP": "cryptomamba_attentionmlp_v",
    "CryptoMamba_AttnGatedMLP": "cryptomamba_attngatedmlp_v",
    "CryptoMamba_Fourier": "cryptomamba_fouriermlp_v",
    "CryptoMamba_TCN": "cryptomamba_tcn_v",
    "cmamba": "cmamba",
    "lstm": "lstm",
    "gru": "gru",
    "bilstm": "bilstm"
}

# Sidebar with reversed order
st.sidebar.header("⚙ Model Controls")
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    sorted(file_to_metrics_key.keys(), reverse=True)  # reversed order
)
show_comparison = st.sidebar.checkbox("Show Full Comparison")
st.sidebar.markdown("---")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
file_path = os.path.join(prediction_folder, f"{model_choice}_predictions.csv")
try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

required_columns = {"Date", "Actual", "Predicted"}
if not required_columns.issubset(df.columns):
    st.error("CSV file must contain: Date, Actual, Predicted columns.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# --------------------------------------------------
# FUNCTION TO PLOT SECTION
# --------------------------------------------------
def plot_section(data, title):
    st.subheader(title)
    fig = px.line(
        data,
        x="Date",
        y=["Actual", "Predicted"],
        labels={"value": "Price (USD)", "variable": "Series", "Date": "Date"},
        template=template
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📋 Actual vs Predicted BTC Price (USD)")
    st.dataframe(data[["Date", "Actual", "Predicted"]], use_container_width=True)
    st.markdown("---")

# --------------------------------------------------
# DISPLAY TRAIN, VALIDATION, TEST
# --------------------------------------------------
plot_section(train_df, f"📊 {model_choice} - Train Set (70%)")
plot_section(val_df, f"📊 {model_choice} - Validation Set (15%)")
plot_section(test_df, f"📊 {model_choice} - Test Set (15%)")

# Full Dataset
st.subheader(f"📈 {model_choice} - Complete Dataset")
fig_full = px.line(
    df,
    x="Date",
    y=["Actual", "Predicted"],
    labels={"value": "Price (USD)", "variable": "Series", "Date": "Date"},
    template=template
)
st.plotly_chart(fig_full, use_container_width=True)
st.markdown("### 📋 Complete Actual vs Predicted BTC Price Table")
st.dataframe(df, use_container_width=True)
st.markdown("---")

# --------------------------------------------------
# DISPLAY METRICS
# --------------------------------------------------
st.subheader("📌 Model Evaluation Metrics")
metrics_key = file_to_metrics_key.get(model_choice)

if metrics_key in model_metrics:
    metrics = model_metrics[metrics_key]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE (USD)", f"{metrics['RMSE']:.2f}")
    col2.metric("MAE (USD)", f"{metrics['MAE']:.2f}")
    col3.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")
    col4.metric("R² Score", f"{metrics['R2']:.4f}")

    best_rmse = min([m["RMSE"] for m in model_metrics.values()])
    if metrics["RMSE"] == best_rmse:
        st.success("🏆 Best Performing Model (Lowest RMSE)")
else:
    st.warning("Metrics not available for this model.")

st.markdown("---")

# --------------------------------------------------
# FULL COMPARISON
# --------------------------------------------------
if show_comparison:
    st.header("🏆 Comparative Performance of All Models")
    comparison_df = pd.DataFrame([
        {
            "Model": model.upper(),
            "RMSE": values["RMSE"],
            "MAE": values["MAE"],
            "MAPE (%)": values["MAPE"],
            "R2 Score": values["R2"]
        } for model, values in model_metrics.items()
    ]).sort_values("RMSE")

    st.dataframe(comparison_df, use_container_width=True)

    best_model = comparison_df.iloc[0]
    st.success(f"🥇 Best Model: {best_model['Model']} (RMSE: {best_model['RMSE']})")

    fig_bar = px.bar(
        comparison_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        template=template,
        title="RMSE Comparison Across Models"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------
# DOWNLOAD BUTTON
# --------------------------------------------------
st.download_button(
    label="📥 Download Selected Model Predictions",
    data=df.to_csv(index=False),
    file_name=f"{model_choice}_predictions.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(
    "Developed By Muhammad Younis, Haseeb and Waqar Ahmed "
    "(BE Computer Systems Engineering, Sukkur IBA University) "
    "| Deep Learning for Financial Time Series Forecasting"
)