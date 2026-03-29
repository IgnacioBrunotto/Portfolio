# Sales Forecasting with XGBoost & LightGBM

Predicting monthly retail sales using gradient boosting models with hyperparameter optimization via Optuna.

---

## Problem Statement

Given 4 years of historical monthly sales data from a retail business, the goal is to forecast future sales with the highest possible accuracy. This is a supervised regression problem framed as a time series task, where future values are predicted using lag features and rolling statistics derived from past observations.

---

## Dataset

| Field | Detail |
|---|---|
| Source | `sales_data.csv` |
| Granularity | Monthly aggregated sales |
| Time range | ~4 years |
| Target variable | `Sales` (total revenue per month) |

---

## Methodology

### Feature Engineering
The raw date column was transformed into predictive features:

- **Date features:** year, month, quarter
- **Lag features:** lag_1, lag_2, lag_3, lag_6, lag_9, lag_12
- **Rolling statistics:** rolling mean (3, 6, 12 months), rolling std (3 months)
- **Year-over-year growth:** percentage change vs same month prior year
- **Expanding mean:** cumulative historical average

### Train / Test Split
A time-based split was used (no random shuffling) to respect the temporal order of the data:
- **Train:** all data before January 2017
- **Test:** January 2017 onwards

### Hyperparameter Optimization
Both models were tuned using **Optuna** with **TimeSeriesSplit (5 folds)** cross-validation, optimizing for Mean Absolute Error (MAE). Optuna uses Bayesian optimization to efficiently search the parameter space, outperforming grid search in both speed and quality.

### Models
| Model | Key parameters tuned |
|---|---|
| XGBoost | n_estimators, learning_rate, max_depth, subsample, colsample_bytree, min_child_weight, gamma |
| LightGBM | n_estimators, learning_rate, max_depth, num_leaves, subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda |

---

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | — | — | ~0.82 |
| LightGBM | — | — | ~0.82 |

> Both models achieved an R² of ~0.82, meaning they explain 82% of the variance in monthly sales — a strong result given the small dataset size (~48 monthly observations).

### Predictions vs Actual

The chart below shows both models' predictions against real sales values on the test set:

![Predictions vs Actual](predictions.png)

### 12-Month Forecast

Using recursive forecasting (each prediction becomes input for the next), both models projected sales 12 months into the future:

![Forecast](forecast.png)

---

## Forecasting Approach

Since the models use lag features, future predictions require a **recursive strategy**: predict one month at a time and feed each prediction back as a lag for the next step. This is implemented in the `forecast_future()` function in the notebook.

---

## Tech Stack

- **Python 3.12**
- `pandas`, `numpy` — data manipulation
- `xgboost`, `lightgbm` — gradient boosting models
- `optuna` — hyperparameter optimization
- `scikit-learn` — cross-validation, metrics
- `matplotlib`, `seaborn` — visualization

---

## How to Run

```bash
pip install xgboost lightgbm optuna scikit-learn pandas numpy matplotlib seaborn
```

Open `forecasting.ipynb` and run all cells in order.
