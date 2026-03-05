# Explainable AI-Based Traffic Congestion Prediction Using SHAP Analysis

Aplikasi ini adalah implementasi penelitian untuk **prediksi jumlah kendaraan per jam** menggunakan:
- **XGBoost Regressor** (model utama)
- **Random Forest Regressor** (pembanding)
- **SHAP** untuk interpretasi global dan lokal

## 1) Download Dataset Kaggle
Gunakan Kaggle CLI:

```bash
kaggle datasets download fedesoriano/traffic-prediction-dataset
unzip traffic-prediction-dataset.zip -d data
```

Pastikan file CSV tersedia lalu upload ke aplikasi Streamlit.

## 2) Fitur Penelitian
Input features yang digunakan:
- Hour
- Day
- Month
- Weekday / Weekend
- Junction
- Lag features: `vehicles t-1`, `t-2`, `t-24`

## 3) Arsitektur Sistem
Dataset → Feature Engineering → Train/Test Split → XGBoost Training → Evaluation (MAE, RMSE) → SHAP Analysis → Interpretation

## 4) Explainable AI dengan SHAP
Aplikasi menyediakan:
- SHAP Summary Plot (global importance)
- SHAP Dependence Plot (fitur Hour/jam sibuk)
- Local explanation (waterfall plot per observasi)

## 5) Penguatan Akademik
Termasuk:
- Perbandingan XGBoost vs Random Forest
- Analisis jam sibuk paling berpengaruh (mean |SHAP Hour|)
- Analisis performa per junction
- Visualisasi pola harian lalu lintas

## Menjalankan aplikasi
```bash
pip install -r requirements.txt
streamlit run app.py
```
