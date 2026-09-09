# Customer Intelligence System — Classification, Ensemble Learning & Clustering on Country Data

## Overview

This project builds an end-to-end "Customer Intelligence System" on the Kaggle **Unsupervised Learning on Country Data** dataset. Countries are treated as "customers": they're first segmented using unsupervised clustering, and those segments are then used as labels to train supervised classifiers that predict a country's segment from its raw socioeconomic features.

## Dataset

- **File:** `Country-data.csv`
- **Shape:** 167 rows × 10 columns
- **Columns:** `country`, `child_mort`, `exports`, `health`, `imports`, `income`, `inflation`, `life_expec`, `total_fer`, `gdpp`
- No missing values or duplicates after cleaning.

## Pipeline Steps

1. **Setup** — Install and import pandas, numpy, matplotlib, seaborn, scikit-learn, and xgboost.
2. **Load & Inspect** — Load the CSV, check shape, dtypes, and summary statistics.
3. **Cleaning** — Normalize column names, drop duplicates, coerce numeric columns, fill any missing values with the column median.
4. **EDA** — Correlation heatmap across all numeric features, plus boxplots per feature to spot outliers.
5. **Feature Scaling** — Standardize all numeric features with `StandardScaler` ahead of clustering/classification.
6. **K-Means Clustering** — Elbow method over k = 2–10 to select the number of clusters; final model trained with k = 3.
7. **DBSCAN** — Density-based clustering as an alternative segmentation, for comparison.
8. **PCA Visualization** — Project scaled features to 2D and plot K-Means clusters.
9. **Cluster Profiling** — Mean feature values per cluster to characterize each segment.
10. **Classification** — Use the K-Means cluster labels as targets; train Random Forest and XGBoost classifiers, plus a soft-voting ensemble, to predict a country's segment from its features.
11. **Feature Importance** — Random Forest importances to see which features drive segment membership.
12. **Final Insights** — Human-readable summary of what each cluster represents.

## Results

### Clustering
- **K-Means (k=3):** Silhouette score ≈ 0.283
- **DBSCAN (eps=1.5, min_samples=5):** 137 core points in one cluster, 30 points labeled as noise (-1)

### Cluster Profiles (K-Means)

| Cluster | child_mort | income | life_expec | gdpp | Interpretation |
|---|---|---|---|---|---|
| 0 | 5.00 | 45,672 | 80.13 | 42,494 | Developed countries |
| 1 | 92.96 | 3,942 | 59.19 | 1,922 | Least developed — highest aid priority |
| 2 | 21.93 | 12,306 | 72.81 | 6,486 | Developing countries |

### Classification (predicting K-Means segment)

| Model | Accuracy |
|---|---|
| Random Forest | 1.00 |
| XGBoost | 0.94 |
| Voting Ensemble (soft) | 0.97 |

## Key Takeaways

- The three K-Means clusters map cleanly onto **developed**, **developing**, and **least-developed** country groupings based on child mortality, income, life expectancy, and GDP per capita.
- Cluster 1 (least developed — e.g. Burundi, Liberia, Niger) has the highest child mortality and lowest income/GDP per capita, and should be prioritized for aid.
- All three classifiers predicted cluster membership from raw features with high accuracy, confirming the clusters are well separated and can be used to score new/unseen countries.
- DBSCAN found a much coarser structure (one dense cluster + noise points) compared to K-Means's three balanced segments, making K-Means the more useful choice here.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## How to Run

1. Place `Country-data.csv` in the working directory.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`
3. Run the notebook cells sequentially from top to bottom.
