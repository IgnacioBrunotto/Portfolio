# Demand Forecasting — Predicción de Demanda Retail

> Pipeline de forecasting end-to-end comparando **Prophet**, **LightGBM** y **XGBoost** sobre 76.000 transacciones retail en 5 tiendas y 20 SKUs.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-1.1-0068c9)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-3daf4e)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-ec6c12)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn&logoColor=white)

🇬🇧 [English version](README.md)

---

## Resultados principales

| Modelo | MAE | RMSE | MAPE | Ljung-Box |
|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | **193.6** | **243.1** | **1.99%** | p = 0.94 — ruido blanco ✅ |
| XGBoost | 200.3 | 253.4 | 2.03% | p = 0.35 — ruido blanco ✅ |
| Prophet | 255.5 | 322.4 | 2.49% | p = 0.33 — ruido blanco ✅ |

Los tres modelos superan el test de Ljung-Box → los residuos son **ruido blanco** (no queda estructura temporal explotable).

---

## Tabla de contenidos

- [Descripción del problema](#descripción-del-problema)
- [Dataset](#dataset)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Metodología](#metodología)
- [Resultados de los modelos](#resultados-de-los-modelos)
- [Análisis de residuos](#análisis-de-residuos)
- [Cómo reproducir](#cómo-reproducir)

---

## Descripción del problema

Predecir la demanda con precisión es crítico para optimizar inventario, planificar promociones y prevenir quiebres de stock. Este proyecto construye un pipeline completo de ML que:

1. Identifica los principales drivers de demanda mediante EDA (promociones, epidemias, clima, estacionalidad).
2. Genera features temporales: lags, medias móviles, términos de interacción y features de calendario.
3. Entrena y compara tres familias de modelos, cada una con tuning de hiperparámetros y validación temporal.
4. Valida la calidad del forecast mediante análisis de residuos (ACF, Ljung-Box, tests de normalidad).

---

## Dataset

| Atributo | Valor |
|---|---|
| Registros | 76.000 |
| Variables | 16 |
| Período | Ene 2022 – Ene 2024 (760 puntos diarios tras agregación) |
| Tiendas | 5 (S001–S005) |
| SKUs | 20 (P0001–P0020) |
| Categorías | Groceries · Clothing · Electronics · Toys · Furniture |
| Valores nulos | Ninguno |
| Variable objetivo | `Demand` — unidades diarias totales vendidas |

**Variables principales:**

| Variable | Tipo | Descripción |
|---|---|---|
| `Price` / `Competitor Pricing` | numérica | Precio propio y benchmark competitivo |
| `Discount` / `Promotion` | numérica / binaria | Descuento aplicado y flag de promoción activa |
| `Inventory Level` | numérica | Stock disponible al inicio del día |
| `Weather Condition` | categórica | Sunny · Cloudy · Rainy · Snowy |
| `Seasonality` | categórica | Spring · Summer · Autumn · Winter |
| `Epidemic` | binaria | Indicador de evento epidémico (presente en ~20% de los datos) |

**Hallazgos clave del EDA:**

| Driver | Efecto sobre la demanda |
|---|---|
| Promoción activa | **+30%** demanda media |
| Evento epidémico | **−38%** demanda media |
| Categoría (Groceries vs Furniture) | 121 vs 74 unidades medias |
| Verano vs Primavera | Pico vs valle estacional |

---

## Estructura del repositorio

```
demand_forecasting/
│
├── demand_forecasting.csv       # Dataset original
├── Demand_Forecasting.ipynb     # Notebook de EDA
├── Modelado.ipynb               # Pipeline de modelado
├── EDA_Informe.md               # Informe detallado de hallazgos del EDA
├── README.md                    # Versión en inglés
└── README_ES.md                 # Este archivo (español)
```

---

## Metodología

### 1. Decisiones de preprocesamiento

| Decisión | Justificación |
|---|---|
| Excluir `Units Sold` | Correlación 0.83 con `Demand` → data leakage |
| Excluir `Units Ordered` | Posible look-ahead bias |
| Fusionar `Discount` + `Promotion` → `Promotion_Ratio` | Multicolinealidad; el ratio captura la intensidad |
| Crear `price_vs_category_mean` | El precio global pierde señal sin contexto de categoría |
| `epidemic_lag_7` en lugar de `Epidemic` cruda | Elimina look-ahead bias en la señal de epidemia |

### 2. Feature engineering

```python
# Lags sobre la demanda diaria agregada
lag_1, lag_7, lag_14, lag_30

# Estadísticas móviles
rolling_mean_7,  rolling_mean_14,  rolling_mean_30
rolling_std_7,   rolling_std_14,   rolling_std_30

# Interacciones de negocio
Promotion_Ratio          # Intensidad media diaria de promociones
promo_x_epidemic         # Efecto de la promoción durante epidemia
epidemic_lag_7           # Epidemia con retardo (sin leakage)
price_vs_category_mean   # Posicionamiento relativo de precio

# Calendario
month, quarter, weekofyear, is_weekend
```

### 3. División temporal

```
Train  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Ene 2022 → Oct 2023  (~22 meses, 639 días)
Test   ░░░░░░░░░░░░░░░░░░░░░░░░  Nov 2023 → Ene 2024  (~3 meses,   91 días)
```

Se utiliza `TimeSeriesSplit` (5 folds) durante el tuning — ningún dato futuro filtra al entrenamiento.

### 4. Tuning de hiperparámetros

| Modelo | Estrategia | Espacio de búsqueda |
|---|---|---|
| **Prophet** | Grid search manual + CV temporal | `changepoint_prior_scale` × `seasonality_prior_scale` × `seasonality_mode` (18 combinaciones) |
| **LightGBM** | `RandomizedSearchCV`, 40 iteraciones, `TimeSeriesSplit` | 9 hiperparámetros |
| **XGBoost** | `RandomizedSearchCV`, 40 iteraciones, `TimeSeriesSplit` | 9 hiperparámetros |

---

## Resultados de los modelos

### Prophet

Recibe regressors externos (`Promotion_Ratio`, `epidemic_lag_7`, `Avg_Discount`, `Avg_Price`, `promo_x_epidemic`) y aplica descomposición automática de estacionalidad semanal y anual. Los intervalos de confianza al 80% brindan una medida interpretable de incertidumbre — útil para presentaciones a stakeholders.

### LightGBM

Obtiene el menor MAE y MAPE. Los features de lag y medias móviles dominan la importancia — consistente con la estructura de autocorrelación de la demanda. Epidemia y promoción figuran entre los principales drivers de negocio.

### XGBoost

Rendimiento comparable a LightGBM con un RMSE ligeramente mayor, mostrando mayor resistencia a picos de outliers individuales. Los patrones de importancia de features son similares entre ambos modelos de gradient boosting.

---

## Análisis de residuos

Se realizó un análisis riguroso de residuos para validar la calidad del forecast más allá de las métricas puntuales.

**Test de Ljung-Box (lag = 10):**

| Modelo | Estadístico | p-valor | Veredicto |
|:---:|:---:|:---:|:---:|
| Prophet | 11.34 | 0.33 | ✅ Ruido blanco |
| LightGBM | 4.21 | 0.94 | ✅ Ruido blanco |
| XGBoost | 11.07 | 0.35 | ✅ Ruido blanco |

Todos los modelos superan el test con α = 0.05 — no queda autocorrelación significativa en los residuos.

**Distribución de residuos:**

| Modelo | Media | Std | Sesgo | Shapiro-Wilk p | Normalidad |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Prophet | −90.6 | 248.7 | −0.42 | 0.37 | ✅ Normal |
| LightGBM | −28.0 | 255.5 | −0.21 | 0.92 | ✅ Normal |
| XGBoost | −20.5 | 290.0 | +0.94 | 0.00 | ❌ No normal |

LightGBM presenta la mejor combinación: menor sesgo (media ≈ −28), distribución simétrica y normalidad confirmada.

---

## Cómo reproducir

### 1. Clonar el repositorio

```bash
git clone https://github.com/IgnacioBrunotto/Portfolio.git
cd Portfolio/demand_forecasting
```

### 2. Crear entorno e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scikit-learn \
            lightgbm xgboost prophet statsmodels scipy jupyter
```

### 3. Ejecutar los notebooks en orden

| # | Notebook | Propósito |
|---|---|---|
| 1 | `Demand_Forecasting.ipynb` | EDA completo |
| 2 | `Modelado.ipynb` | Feature engineering, entrenamiento y evaluación |

> **Nota:** El grid search de Prophet ejecuta 18 configuraciones × CV temporal — esperar 5–10 minutos según el hardware.

---

## Stack tecnológico

```
pandas · numpy · matplotlib · seaborn
scikit-learn  →  RandomizedSearchCV · TimeSeriesSplit · MAE · RMSE · MAPE
prophet       →  regressors · grid search · intervalos de confianza
lightgbm      →  LGBMRegressor
xgboost       →  XGBRegressor
statsmodels   →  acorr_ljungbox · plot_acf
scipy         →  shapiro · probplot
```

---

*Proyecto de portfolio de ciencia de datos — forecasting de demanda retail end-to-end.*
