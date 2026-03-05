# Explainable AI-Based Traffic Congestion Prediction (SHAP)

Aplikasi ini melakukan prediksi **tingkat kemacetan lalu lintas** (`congestion_index`) menggunakan model Machine Learning dan menjelaskan hasil prediksi menggunakan **SHAP**.

## Fitur
- Prediksi kemacetan berbasis input kondisi lalu lintas dan cuaca.
- Evaluasi model otomatis (MAE, R²).
- **Global explainability**: SHAP summary plot.
- **Local explainability**: kontribusi tiap fitur untuk satu prediksi spesifik.

## Menjalankan aplikasi
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Variabel input
- `hour`
- `day_of_week`
- `rain_mm`
- `temperature_c`
- `holiday`
- `event_nearby`
- `road_incident`
- `avg_speed_kmh`
- `traffic_volume`

## Catatan
Dataset pada contoh ini bersifat **sintetis** (dibangkitkan otomatis) agar aplikasi dapat langsung dijalankan. Untuk penggunaan nyata, ganti bagian generator data dengan data riil dari sensor/ATCS/GPS/API trafik.
