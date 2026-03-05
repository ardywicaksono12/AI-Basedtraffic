import io
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


st.set_page_config(page_title="Explainable AI-Based Traffic Congestion Prediction", layout="wide")


@dataclass
class ModelResults:
    xgb_model: XGBRegressor
    rf_model: RandomForestRegressor
    X_test: pd.DataFrame
    y_test: pd.Series
    xgb_pred: np.ndarray
    rf_pred: np.ndarray
    metrics: pd.DataFrame


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if low in {"datetime", "date_time", "date"}:
            rename_map[col] = "DateTime"
        elif low == "vehicles":
            rename_map[col] = "Vehicles"
        elif low == "junction":
            rename_map[col] = "Junction"
    out = df.rename(columns=rename_map).copy()
    required = {"DateTime", "Vehicles", "Junction"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    return out.sort_values(["Junction", "DateTime"]).reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Hour"] = df["DateTime"].dt.hour
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["Weekday"] = (df["DateTime"].dt.dayofweek < 5).astype(int)
    df["Weekend"] = (df["DateTime"].dt.dayofweek >= 5).astype(int)

    for lag in [1, 2, 24]:
        df[f"Vehicles_lag_{lag}"] = df.groupby("Junction")["Vehicles"].shift(lag)

    return df.dropna().reset_index(drop=True)


def train_models(df_feat: pd.DataFrame) -> ModelResults:
    features = [
        "Hour",
        "Day",
        "Month",
        "Weekday",
        "Weekend",
        "Junction",
        "Vehicles_lag_1",
        "Vehicles_lag_2",
        "Vehicles_lag_24",
    ]
    train_parts = []
    test_parts = []
    for _, group in df_feat.groupby("Junction", sort=False):
        if len(group) < 2:
            raise ValueError("Setiap junction harus memiliki minimal 2 baris untuk train/test split berbasis waktu.")
        split_idx = int(len(group) * 0.8)
        split_idx = min(max(split_idx, 1), len(group) - 1)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    X_train = train_df[features]
    y_train = train_df["Vehicles"]
    X_test = test_df[features]
    y_test = test_df["Vehicles"]

    xgb_model = XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    metrics = pd.DataFrame(
        {
            "Model": ["XGBoost", "Random Forest"],
            "MAE": [
                mean_absolute_error(y_test, xgb_pred),
                mean_absolute_error(y_test, rf_pred),
            ],
            "RMSE": [
                np.sqrt(mean_squared_error(y_test, xgb_pred)),
                np.sqrt(mean_squared_error(y_test, rf_pred)),
            ],
        }
    )

    return ModelResults(xgb_model, rf_model, X_test, y_test, xgb_pred, rf_pred, metrics)


def fig_to_streamlit(fig):
    st.pyplot(fig, clear_figure=True)


st.title("Explainable AI-Based Traffic Congestion Prediction Using SHAP Analysis")
st.markdown(
    """
Arsitektur: **Dataset → Feature Engineering → Train/Test Split → XGBoost + Random Forest → Evaluation (MAE, RMSE) → SHAP Analysis → Interpretation**
"""
)

st.sidebar.header("Data")
st.sidebar.write("Unduh data dengan perintah:")
st.sidebar.code("kaggle datasets download fedesoriano/traffic-prediction-dataset")
uploaded = st.sidebar.file_uploader("Upload CSV hasil ekstraksi Kaggle", type=["csv"])

if uploaded is None:
    st.info("Silakan upload file CSV dari dataset Kaggle untuk memulai analisis.")
    st.stop()

raw = pd.read_csv(uploaded)
try:
    data = standardize_columns(raw)
except Exception as exc:
    st.error(str(exc))
    st.stop()

feat = feature_engineering(data)
results = train_models(feat)

st.subheader("Preview Data")
st.dataframe(feat.head(10), use_container_width=True)

st.subheader("Evaluasi Model")
st.dataframe(results.metrics, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=np.arange(len(results.y_test)), y=results.y_test.values, label="Actual", ax=ax)
    sns.lineplot(x=np.arange(len(results.xgb_pred)), y=results.xgb_pred, label="XGB Pred", ax=ax)
    ax.set_title("Actual vs XGBoost Prediction")
    ax.set_xlabel("Test Index")
    ax.set_ylabel("Vehicles")
    fig_to_streamlit(fig)

with col2:
    daily_pattern = feat.copy()
    daily_pattern["Date"] = daily_pattern["DateTime"].dt.date
    day_profile = daily_pattern.groupby("Hour")["Vehicles"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=day_profile, x="Hour", y="Vehicles", ax=ax)
    ax.set_title("Visualisasi Pola Harian (Rata-rata Kendaraan/Jam)")
    fig_to_streamlit(fig)

st.subheader("SHAP Analysis (XGBoost)")
explainer = shap.TreeExplainer(results.xgb_model)
shap_values = explainer.shap_values(results.X_test)

fig, ax = plt.subplots(figsize=(9, 5))
shap.summary_plot(shap_values, results.X_test, show=False)
fig = plt.gcf()
st.pyplot(fig, clear_figure=True)

st.markdown("#### SHAP Dependence Plot: Pengaruh Jam (Hour)")
fig, ax = plt.subplots(figsize=(8, 5))
shap.dependence_plot("Hour", shap_values, results.X_test, ax=ax, show=False)
st.pyplot(fig, clear_figure=True)

st.markdown("#### Local Explanation")
idx = st.slider("Pilih index observasi test", 0, len(results.X_test) - 1, 0)
local_exp = shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=results.X_test.iloc[idx],
    feature_names=results.X_test.columns,
)
fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(local_exp, max_display=10, show=False)
st.pyplot(plt.gcf(), clear_figure=True)

st.subheader("Analisis Tambahan Akademik")
peak = pd.DataFrame({"Hour": results.X_test["Hour"], "Abs_SHAP_Hour": np.abs(shap_values[:, 0])})
peak_hour_impact = peak.groupby("Hour")["Abs_SHAP_Hour"].mean().sort_values(ascending=False)
st.write("Jam dengan pengaruh terbesar (berdasarkan mean |SHAP Hour|):")
st.dataframe(peak_hour_impact.reset_index().rename(columns={"Abs_SHAP_Hour": "MeanAbsSHAP"}))

junction_eval = pd.DataFrame(
    {
        "Junction": results.X_test["Junction"].values,
        "Actual": results.y_test.values,
        "Pred": results.xgb_pred,
    }
)
mae_per_junction = (
    junction_eval.groupby("Junction").apply(lambda g: mean_absolute_error(g["Actual"], g["Pred"])).reset_index(name="MAE")
)
st.write("Analisis per Junction (MAE XGBoost):")
st.dataframe(mae_per_junction, use_container_width=True)

csv_buffer = io.StringIO()
results.metrics.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download evaluasi model (CSV)",
    data=csv_buffer.getvalue(),
    file_name="model_comparison_metrics.csv",
    mime="text/csv",
)
