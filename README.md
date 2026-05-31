# 🚗 Tesla EV Delivery Forecasting — ML Pipeline
**Celebal Technologies | Data Science Internship | Assignment 2**

---

## Overview
Built an end-to-end machine learning pipeline to forecast Tesla vehicle deliveries using 10 years of historical data (2015–2025) across 5 models and 4 regions.

**Business Question:** Can we predict Tesla deliveries for the next 6 months?

---

## Dataset
| Property | Detail |
|---|---|
| Source | Kaggle — Tesla EA Deliveries and Production Data |
| Records | 2,640 rows |
| Period | 2015–2025 |
| Target | Estimated_Deliveries |

---

## What Was Applied

✅ **Data Cleaning** — handled missing values, duplicates, outlier capping via IQR, fixed data types

✅ **EDA** — analysed yearly trends, seasonal patterns, regional distribution, correlation heatmap

✅ **Encoding** — One-Hot Encoding for Region and Model, Label Encoding for Source_Type

✅ **Feature Scaling** — StandardScaler applied inside Pipeline on training data only

✅ **Feature Engineering** — lag features (1, 3, 12 months), rolling mean and std, quarter flags, interaction features

✅ **Data Leakage Prevention** — removed CO2_Saved_tons (calculated from target), scaled inside Pipeline to avoid test data contamination

✅ **Regression Models** — trained Linear Regression, Ridge (L2), and Lasso (L1) inside sklearn Pipelines

✅ **Evaluation** — MAE, RMSE, R², MAPE across all three models

✅ **Cross Validation** — TimeSeriesSplit (5 folds) to respect time order

✅ **Hyperparameter Tuning** — GridSearchCV over alpha values for Ridge and Lasso

✅ **Stationarity Testing** — ADF Test using statsmodels

✅ **Forecasting** — predicted next 6 months using best tuned model

✅ **Chronological Split** — Train: 2015–2022 | Test: 2023–2025

---

## Tech Stack
```
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Statsmodels
```

---

## Results

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 345 | 422 | 0.9869 | 3.6% |
| Ridge Regression | 345 | 422 | 0.9869 | 3.6% |
| Lasso Regression | 344 | 421 | 0.9869 | 3.6% |

---

## Run It
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
jupyter notebook tesla_ml_project.ipynb
```

---

