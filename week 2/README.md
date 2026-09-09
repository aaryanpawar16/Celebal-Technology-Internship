# Tesla Deliveries & Production — ML Pipeline (2015–2025)

**Author:** Aaryan Pawar
**Assignment:** Week 2 — End-to-End Machine Learning Pipeline
**Dataset:** Tesla Deliveries and Production Data (2015–2025)

## Overview

This project builds a complete, end-to-end machine learning pipeline on a synthetic Tesla delivery and production dataset spanning 2015–2025. It covers everything from raw data inspection through model comparison and forecasting insight.

## Dataset

- **File:** `tesla_deliveries_dataset_2015_2025.csv`
- **Shape:** 2,640 rows × 12 columns
- **Columns:** `Year`, `Month`, `Region`, `Model`, `Estimated_Deliveries`, `Production_Units`, `Avg_Price_USD`, `Battery_Capacity_kWh`, `Range_km`, `CO2_Saved_tons`, `Source_Type`, `Charging_Stations`
- No missing values or duplicate rows.

## Pipeline Steps

1. **Load & Inspect** — Load CSV, check shape, dtypes, and summary statistics.
2. **Data Quality** — Check for nulls and duplicates (none found).
3. **Exploratory Data Analysis** — 5 charts: deliveries by model, deliveries by region, correlation heatmap, production vs. deliveries scatter, and monthly delivery trend.
4. **Feature Engineering** — Label-encode categorical columns (`Region`, `Model`, `Source_Type`); create `Deliveries_Lag1` and `Rolling_Mean_3` features, with NaNs filled using column means.
5. **Train/Test Split** — Chronological 80/20 split (no shuffling) to prevent data leakage.
6. **Linear Regression** — Baseline model trained on 12 features.
7. **5-Fold Cross Validation** — Validates model stability on the training set.
8. **Random Forest + GridSearchCV** — Hyperparameter tuning over `n_estimators` and `max_depth`, with feature importance analysis.
9. **ADF Stationarity Test** — Augmented Dickey-Fuller test on the aggregated monthly delivery series.
10. **Forecast Comparison** — Side-by-side actual vs. predicted table for both models, plus a final metrics summary.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 310.36 | 375.53 | 0.9908 |
| Random Forest (Tuned) | 303.97 | 387.26 | 0.9902 |

- **Best Random Forest params:** `max_depth=10`, `n_estimators=100`
- **5-Fold CV mean R²:** 0.9903 (std dev 0.0007)
- **ADF test:** p-value ≈ 0.0000 → series is stationary
- **Top predictor:** `Production_Units`, consistent with its 0.9942 correlation with `Estimated_Deliveries`

## Key Takeaways

- Production and deliveries are almost perfectly linearly related, so Linear Regression edges out the tuned Random Forest.
- Both models generalize well and show low, consistent error across cross-validation folds.
- The stationarity result supports using the series directly for further time-series modeling without differencing.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

## How to Run

1. Place `tesla_deliveries_dataset_2015_2025.csv` in the working directory.
2. Run the notebook cells sequentially from top to bottom.
