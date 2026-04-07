# 📦 Demand Forecasting — Retail Sales Prediction

> End-to-end forecasting pipeline comparing **Prophet**, **LightGBM** and **XGBoost** on 76,000 retail transactions across 5 stores and 20 SKUs.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-1.1-0068c9)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-3daf4e)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-ec6c12)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn&logoColor=white)

---

## Results at a glance

![Forecast Comparison](images/forecast_comparison.png)

| Model | MAE | RMSE | MAPE | Ljung-Box |
|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | **193.6** | **243.1** | **1.99%** ✅ | p = 0.94 — white noise |
| XGBoost | 200.3 | 253.4 | 2.03% | p = 0.35 — white noise |
| Prophet | 255.5 | 322.4 | 2.49% | p = 0.33 — white noise |

All three models pass the Ljung-Box test → residuals are **white noise** (no exploitable temporal structure left behind).

---

## Table of contents

- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Methodology](#methodology)
- [Model results](#model-results)
- [Residual analysis](#residual-analysis)
- [How to reproduce](#how-to-reproduce)

---

## Problem statement

Forecasting demand accurately is critical for inventory optimization, promotion planning and stockout prevention. This project builds a complete ML pipeline that:

1. Identifies the main demand drivers through EDA (promotions, epidemics, weather, seasonality).
2. Engineers time-aware features: lags, rolling statistics, interaction terms and calendar features.
3. Trains and compares three model families — each with proper temporal hyperparameter tuning.
4. Validates forecast quality through residual analysis (ACF, Ljung-Box, normality tests).

---

## Dataset

| Attribute | Value |
|---|---|
| Records | 76,000 |
| Features | 16 |
| Period | Jan 2022 – Jan 2024 (760 daily points after aggregation) |
| Stores | 5 (S001–S005) |
| SKUs | 20 (P0001–P0020) |
| Categories | Groceries · Clothing · Electronics · Toys · Furniture |
| Missing values | None |
| Target | `Demand` — total daily units sold |

**Key variables:**

| Variable | Type | Description |
|---|---|---|
| `Price` / `Competitor Pricing` | numeric | Own price and competitive benchmark |
| `Discount` / `Promotion` | numeric / binary | Discount rate and active promotion flag |
| `Inventory Level` | numeric | Opening stock at day start |
| `Weather Condition` | categorical | Sunny · Cloudy · Rainy · Snowy |
| `Seasonality` | categorical | Spring · Summer · Autumn · Winter |
| `Epidemic` | binary | Epidemic event flag (present in ~20% of data) |

**Key EDA findings:**

| Driver | Effect on demand |
|---|---|
| Active promotion | **+30%** average demand |
| Epidemic event | **−38%** average demand |
| Category (Groceries vs Furniture) | 121 vs 74 mean units |
| Summer vs Spring | Peak vs valley seasonality |

---

## Repository structure

```
demand-forecasting/
│
├── demand_forecasting.csv       # Raw dataset
│
├── Demand_Forecasting.ipynb     # EDA notebook
│   ├── Descriptive statistics
│   ├── Data leakage detection
│   ├── Promotion & epidemic impact analysis
│   ├── Category / region / weather segmentation
│   └── Time series visualizations
│
├── Modelado.ipynb               # Modeling pipeline
│   ├── Preprocessing & feature engineering
│   ├── Temporal train/test split
│   ├── Prophet (grid search + temporal CV)
│   ├── LightGBM (RandomizedSearchCV + TimeSeriesSplit)
│   ├── XGBoost  (RandomizedSearchCV + TimeSeriesSplit)
│   ├── Metric comparison (MAE · RMSE · MAPE)
│   └── Residual analysis (ACF · Ljung-Box · distribution)
│
└── images/                      # Plots used in this README
```

---

## Methodology

### 1. Preprocessing decisions

| Decision | Rationale |
|---|---|
| Excluded `Units Sold` | Correlation 0.83 with `Demand` → data leakage |
| Excluded `Units Ordered` | Potential look-ahead bias |
| Merged `Discount` + `Promotion` → `Promotion_Ratio` | Multicollinearity; ratio captures intensity |
| Created `price_vs_category_mean` | Raw price loses signal without segment context |
| `epidemic_lag_7` instead of raw `Epidemic` | Eliminates look-ahead bias in epidemic signal |

### 2. Feature engineering

```python
# Lag features (over aggregated daily demand)
lag_1, lag_7, lag_14, lag_30

# Rolling statistics
rolling_mean_7,  rolling_mean_14,  rolling_mean_30
rolling_std_7,   rolling_std_14,   rolling_std_30

# Business interactions
Promotion_Ratio          # Daily mean promotion intensity
promo_x_epidemic         # Promotion effect during epidemic
epidemic_lag_7           # Lagged epidemic flag (no leakage)
price_vs_category_mean   # Relative price positioning

# Calendar
month, quarter, weekofyear, is_weekend
```

### 3. Temporal split

```
Train  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Jan 2022 → Oct 2023  (~22 months, 639 days)
Test   ░░░░░░░░░░░░░░░░░░░░░░░░  Nov 2023 → Jan 2024  (~3 months,   91 days)
```

`TimeSeriesSplit` (5 folds) used during tuning — no future data ever leaks into training.

### 4. Hyperparameter tuning

| Model | Strategy | Search space |
|---|---|---|
| **Prophet** | Manual grid search + temporal CV | `changepoint_prior_scale` × `seasonality_prior_scale` × `seasonality_mode` (18 combos) |
| **LightGBM** | `RandomizedSearchCV`, 40 iterations, `TimeSeriesSplit` | 9 hyperparameters |
| **XGBoost** | `RandomizedSearchCV`, 40 iterations, `TimeSeriesSplit` | 9 hyperparameters |

---

## Model results

<img width="2076" height="768" alt="metrics_comparison" src="https://github.com/user-attachments/assets/3589c6c5-c703-4ee5-8023-f12c0bc71a05" />


### Prophet — with confidence intervals

<img width="1917" height="722" alt="prophet_forecast" src="https://github.com/user-attachments/assets/c395eefa-933d-4224-9ee6-7f9f4d3952d4" />


Prophet receives external regressors (`Promotion_Ratio`, `epidemic_lag_7`, `Avg_Discount`, `Avg_Price`, `promo_x_epidemic`) and applies automatic weekly + yearly seasonality decomposition. The 80% confidence intervals provide interpretable uncertainty bounds — useful for stakeholder presentations.

### LightGBM — feature importance

<img width="1324" height="872" alt="feature_importance_lgb" src="https://github.com/user-attachments/assets/0753554e-9d25-4539-9188-2856dbe42fa7" />


LightGBM achieves the lowest MAE and MAPE. Lag features and rolling means dominate — consistent with demand autocorrelation structure. Epidemic and promotion features rank among the top business drivers.

### XGBoost — feature importance

![XGBoost Feature Importance](images/feature_importance_xgb.png)

XGBoost delivers comparable performance to LightGBM with slightly higher RMSE, showing stronger resistance to individual outlier spikes. Feature importance patterns are similar across both gradient boosting models.

---

## Residual analysis

A rigorous residual analysis was performed to validate forecast quality beyond point metrics.

### Residuals over time

<img width="1920" height="1474" alt="residuals_over_time" src="https://github.com/user-attachments/assets/f62a4e61-f00b-4eaf-ad79-d83869e7f547" />


All three models produce residuals centered around zero with no visible trend or cyclic structure — indicating the models captured the underlying demand dynamics.

### ACF of residuals + Ljung-Box test

<img width="2226" height="574" alt="residuals_acf" src="https://github.com/user-attachments/assets/03d8a1aa-b966-4fae-a803-56cd579211c8" />


| Model | Ljung-Box statistic (lag=10) | p-value | Verdict |
|:---:|:---:|:---:|:---:|
| Prophet | 11.34 | 0.33 | ✅ White noise |
| LightGBM | 4.21 | 0.94 | ✅ White noise |
| XGBoost | 11.07 | 0.35 | ✅ White noise |

All models pass at α=0.05 — no significant autocorrelation remains in the residuals.

### Residuals distribution

<img width="2215" height="721" alt="residuals_distribution" src="https://github.com/user-attachments/assets/7c529bea-4ec4-4db6-8f39-7981b536da63" />


| Model | Mean | Std | Skew | Shapiro-Wilk p | Normality |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Prophet | −90.6 | 248.7 | −0.42 | 0.37 | ✅ Normal |
| LightGBM | −28.0 | 255.5 | −0.21 | 0.92 | ✅ Normal |
| XGBoost | −20.5 | 290.0 | +0.94 | 0.00 | ❌ Non-normal |

LightGBM shows the best combination: lowest bias (mean ≈ −28), symmetric distribution, and confirmed normality.

---


## Tech stack

```
pandas · numpy · matplotlib · seaborn
scikit-learn  →  RandomizedSearchCV · TimeSeriesSplit · MAE · RMSE · MAPE
prophet       →  regressors · grid search · confidence intervals
lightgbm      →  LGBMRegressor
xgboost       →  XGBRegressor
statsmodels   →  acorr_ljungbox · plot_acf
scipy         →  shapiro · probplot
```

---

*Data Science portfolio project — end-to-end retail demand forecasting.*
