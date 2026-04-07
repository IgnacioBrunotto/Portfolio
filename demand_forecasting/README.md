# Demand Forecasting — Retail Sales Prediction

> End-to-end forecasting pipeline comparing **Prophet**, **LightGBM** and **XGBoost** on 76,000 retail transactions across 5 stores and 20 SKUs.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-1.1-0068c9)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-3daf4e)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-ec6c12)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn&logoColor=white)

🇪🇸 [Versión en español](README_ES.md)

---

## Results at a glance

| Model | MAE | RMSE | MAPE | Ljung-Box |
|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | **193.6** | **243.1** | **1.99%** | p = 0.94 — white noise ✅ |
| XGBoost | 200.3 | 253.4 | 2.03% | p = 0.35 — white noise ✅ |
| Prophet | 255.5 | 322.4 | 2.49% | p = 0.33 — white noise ✅ |

All three models pass the Ljung-Box test → residuals are **white noise** (no exploitable temporal structure left behind).

---

## Table of contents

- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Methodology](#methodology)
- [Model results](#model-results)
- [Residual analysis](#residual-analysis)
- [Conclusions](#conclusions)
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
demand_forecasting/
│
├── demand_forecasting.csv       # Raw dataset
├── Demand_Forecasting.ipynb     # EDA notebook
├── Modelado.ipynb               # Modeling pipeline
├── EDA_Informe.md               # Detailed EDA findings report
├── README.md                    # This file (English)
└── README_ES.md                 # Spanish version
```

---

## Methodology

### 1. Preprocessing decisions

| Decision | Rationale |
|---|---|
| Excluded `Units Sold` | Correlation 0.83 with `Demand` → data leakage |
| Excluded `Units Ordered` | Potential look-ahead bias |
| Merged `Discount` + `Promotion` → `Promotion_Ratio` | Multicollinearity; ratio captures promotion intensity |
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

### Prophet

Receives external regressors (`Promotion_Ratio`, `epidemic_lag_7`, `Avg_Discount`, `Avg_Price`, `promo_x_epidemic`) and applies automatic weekly + yearly seasonality decomposition. The 80% confidence intervals provide interpretable uncertainty bounds — useful for stakeholder presentations.

### LightGBM

Achieves the lowest MAE and MAPE. Lag features and rolling means dominate — consistent with demand autocorrelation structure. Epidemic and promotion features rank among the top business drivers.

### XGBoost

Delivers comparable performance to LightGBM with slightly higher RMSE, showing stronger resistance to individual outlier spikes. Feature importance patterns are similar across both gradient boosting models.

---

## Residual analysis

A rigorous residual analysis was performed to validate forecast quality beyond point metrics.

**Ljung-Box test (lag = 10):**

| Model | Statistic | p-value | Verdict |
|:---:|:---:|:---:|:---:|
| Prophet | 11.34 | 0.33 | ✅ White noise |
| LightGBM | 4.21 | 0.94 | ✅ White noise |
| XGBoost | 11.07 | 0.35 | ✅ White noise |

All models pass at α = 0.05 — no significant autocorrelation remains in the residuals.

**Residuals distribution:**

| Model | Mean | Std | Skew | Shapiro-Wilk p | Normality |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Prophet | −90.6 | 248.7 | −0.42 | 0.37 | ✅ Normal |
| LightGBM | −28.0 | 255.5 | −0.21 | 0.92 | ✅ Normal |
| XGBoost | −20.5 | 290.0 | +0.94 | 0.00 | ❌ Non-normal |

LightGBM shows the best combination: lowest bias (mean ≈ −28), symmetric distribution, and confirmed normality.

---

## Conclusions

### Business insights

**Promotions are the most powerful and actionable lever.** A consistent +30% average demand uplift across all categories confirms that the business has direct control over the most impactful demand driver. The `promotion × epidemic` interaction shows that promotions remain effective even in adverse scenarios (+18 units), although they do not fully offset the epidemic shock.

**Epidemic events are the most severe negative shock (−38%)** and affect all regions uniformly. The model treats them as an external input with a 7-day lag, which means that in production a real-time early warning system would be needed to feed that feature reliably.

**Category is the strongest segmentation dimension.** Groceries (mean 121) nearly doubles Furniture (74). Pricing, stocking and promotion decisions should be made at category level rather than globally.

### Modeling insights

**LightGBM is the winning model** across all relevant metrics: lowest MAE (193.6), lowest MAPE (1.99%) and residuals closest to pure white noise (Ljung-Box p = 0.94). It would be the primary candidate for a production deployment.

**Lag and rolling features are the most important predictors**, confirming that demand has strong autocorrelation — the best predictor of tomorrow's demand is the demand of the last 7–14 days. This validates temporal feature engineering as the highest-value step in the pipeline.

**Prophet is useful despite being less precise.** Its real advantage lies in interpretability: trend + seasonality decomposition with confidence intervals. For stakeholder presentations or explaining *why* demand is rising or falling, Prophet is the stronger choice.

**XGBoost produces non-normal residuals** (Shapiro-Wilk p ≈ 0), suggesting asymmetric behavior around demand spikes — it tends to underestimate high-demand peaks. In a business where stockouts are costlier than overstock, this is an operational risk worth monitoring.

### Process insights

**Data leakage was the most critical risk.** `Units Sold` has a 0.83 correlation with `Demand` — if not detected and excluded, the model produces artificially perfect validation metrics and fails entirely in production. Catching it during EDA, before any model training, was the most important decision of the project.

**The multicollinearity between `Discount` and `Promotion` was non-obvious.** The cross-analysis revealed near-perfect dependence — including both features adds noise, not information.

**MAPE ~2% on aggregated daily demand is a solid result**, but context matters: this is the sum across ~100 store×product combinations per day, which smooths individual errors. At the single store or SKU level, error would be considerably higher.

### Key takeaway

The gap between a model that "works in a notebook" and one ready for production lies in the details: correct lag construction, strict temporal splits, leakage detection in EDA, and residual analysis to confirm the model learned the signal — not the noise.

---

## How to reproduce

### 1. Clone the repository

```bash
git clone https://github.com/IgnacioBrunotto/Portfolio.git
cd Portfolio/demand_forecasting
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scikit-learn \
            lightgbm xgboost prophet statsmodels scipy jupyter
```

### 3. Run notebooks in order

| # | Notebook | Purpose |
|---|---|---|
| 1 | `Demand_Forecasting.ipynb` | Full EDA |
| 2 | `Modelado.ipynb` | Feature engineering, training & evaluation |

> **Note:** Prophet grid search runs 18 configurations × temporal CV — expect 5–10 minutes depending on hardware.

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
