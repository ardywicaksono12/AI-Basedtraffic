import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="XAI Traffic Congestion Predictor", layout="wide")


@st.cache_data
def generate_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic traffic dataset."""
    rng = np.random.default_rng(random_state)

    hour = rng.integers(0, 24, n_samples)
    day_of_week = rng.integers(0, 7, n_samples)
    rain_mm = np.clip(rng.normal(2.0, 3.0, n_samples), 0, None)
    temperature_c = rng.normal(29, 4, n_samples)
    holiday = rng.choice([0, 1], n_samples, p=[0.9, 0.1])
    event_nearby = rng.choice([0, 1], n_samples, p=[0.85, 0.15])
    road_incident = rng.choice([0, 1], n_samples, p=[0.92, 0.08])
    avg_speed_kmh = np.clip(rng.normal(40, 10, n_samples), 10, 80)
    traffic_volume = np.clip(rng.normal(1200, 350, n_samples), 200, 3000)

    rush_hour = np.where(((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19)), 1, 0)
    weekend = np.where(day_of_week >= 5, 1, 0)

    congestion_index = (
        0.03 * traffic_volume
        + 6.5 * rush_hour
        + 4.0 * rain_mm
        + 10.0 * road_incident
        + 7.5 * event_nearby
        - 0.8 * avg_speed_kmh
        - 2.0 * weekend
        - 1.5 * holiday
        + rng.normal(0, 8, n_samples)
    )
    congestion_index = np.clip(congestion_index, 0, 100)

    return pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "rain_mm": rain_mm,
            "temperature_c": temperature_c,
            "holiday": holiday,
            "event_nearby": event_nearby,
            "road_incident": road_incident,
            "avg_speed_kmh": avg_speed_kmh,
            "traffic_volume": traffic_volume,
            "congestion_index": congestion_index,
        }
    )


@st.cache_resource
def train_model(df: pd.DataFrame):
    features = [
        "hour",
        "day_of_week",
        "rain_mm",
        "temperature_c",
        "holiday",
        "event_nearby",
        "road_incident",
        "avg_speed_kmh",
        "traffic_volume",
    ]

    X = df[features]
    y = df["congestion_index"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return model, explainer, X_test, y_test, mae, r2, shap_values, features


def main():
    st.title("🚦 Explainable AI-Based Traffic Congestion Prediction")
    st.caption("Prediksi kemacetan lalu lintas dengan penjelasan SHAP (global & lokal).")

    data = generate_data()
    model, explainer, X_test, y_test, mae, r2, shap_values, features = train_model(data)

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("R²", f"{r2:.3f}")

    st.subheader("Input Kondisi Lalu Lintas")
    c1, c2, c3 = st.columns(3)
    with c1:
        hour = st.slider("Jam", 0, 23, 8)
        day_of_week = st.selectbox("Hari (0=Senin, 6=Minggu)", list(range(7)), index=0)
        rain_mm = st.slider("Curah hujan (mm)", 0.0, 20.0, 2.0, 0.1)
    with c2:
        temperature_c = st.slider("Suhu (°C)", 15.0, 40.0, 30.0, 0.1)
        holiday = st.selectbox("Hari libur", [0, 1], index=0)
        event_nearby = st.selectbox("Ada event besar di dekat lokasi", [0, 1], index=0)
    with c3:
        road_incident = st.selectbox("Ada kecelakaan/insiden", [0, 1], index=0)
        avg_speed_kmh = st.slider("Kecepatan rata-rata (km/j)", 5.0, 90.0, 35.0, 0.5)
        traffic_volume = st.slider("Volume kendaraan (kendaraan/jam)", 100, 3500, 1200)

    user_input = pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "rain_mm": rain_mm,
                "temperature_c": temperature_c,
                "holiday": holiday,
                "event_nearby": event_nearby,
                "road_incident": road_incident,
                "avg_speed_kmh": avg_speed_kmh,
                "traffic_volume": traffic_volume,
            }
        ]
    )

    prediction = float(model.predict(user_input)[0])
    st.success(f"Prediksi congestion index: **{prediction:.2f}/100**")

    st.subheader("Global Explainability (SHAP)")
    fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig_summary, clear_figure=True)

    st.subheader("Local Explainability untuk input saat ini")
    local_shap_values = explainer.shap_values(user_input)
    contribution = pd.DataFrame(
        {
            "feature": features,
            "shap_value": local_shap_values[0],
            "feature_value": user_input.iloc[0].values,
        }
    ).sort_values("shap_value", key=np.abs, ascending=False)

    st.dataframe(contribution, use_container_width=True)

    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in contribution["shap_value"]]
    ax_bar.barh(contribution["feature"], contribution["shap_value"], color=colors)
    ax_bar.axvline(0, color="black", linewidth=1)
    ax_bar.set_xlabel("SHAP Value")
    ax_bar.set_ylabel("Feature")
    ax_bar.set_title("Kontribusi fitur terhadap prediksi")
    plt.gca().invert_yaxis()
    st.pyplot(fig_bar, clear_figure=True)

    st.info(
        "Interpretasi cepat: SHAP positif mendorong nilai kemacetan naik, "
        "SHAP negatif menurunkan prediksi kemacetan."
    )


if __name__ == "__main__":
    main()
