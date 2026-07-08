# Credit Card Fraud Detection

> **End-to-end fraud detection ML pipeline for severely imbalanced data**  
> 4 algorithms · F2-score + cost-sensitive threshold · SHAP interpretability · MLflow tracking · Business ROI analysis

---

## Overview

Detect fraudulent credit card transactions from the **Sparkov synthetic dataset** (~1.3M training, ~555K test transactions). The project demonstrates advanced Data Science skills specifically relevant to imbalanced classification:

- **Class imbalance**: ~0.57% fraud (~1:175 ratio)
- **Primary metric**: PR-AUC (not accuracy or ROC-AUC)
- **Threshold optimization**: F2-score (recall weighted 2×) + cost-sensitive (FP=$10, FN=$500)
- **Geospatial feature**: Haversine distance between customer and merchant

---

## Project Structure

```
credit-card-fraud-detection/
│
├── data/                              # Input data (not committed)
│   ├── fraudTrain.csv                 # ~1.3M training transactions
│   └── fraudTest.csv                  # ~555K test transactions
│
├── notebooks/                         # 📓 Data Scientist workflow (6 notebooks)
│   ├── 01_EDA.ipynb                   # Class imbalance + temporal + geo + category analysis
│   ├── 02_Feature_Engineering.ipynb   # Haversine + log transforms + time features
│   ├── 03_Imbalance_Strategies.ipynb  # ⭐ class_weight vs SMOTE vs undersampling
│   ├── 04_Model_Experiments.ipynb     # 4 algorithms + MLflow + PR/ROC comparison
│   ├── 05_Model_Interpretation.ipynb  # SHAP beeswarm + waterfall + dependence
│   └── 06_Business_Report.ipynb       # Cost matrix + ROI + risk tier segmentation
│
├── src/                               # ⚙️ Production-ready source code
│   ├── data/
│   │   ├── loader.py                  # Stratified subsampling + schema validation
│   │   └── features.py               # Haversine, datetime, log transforms
│   ├── models/
│   │   ├── trainers.py               # 4 classifiers + F2 + cost threshold optimization
│   │   └── evaluation.py             # 12 fraud metrics + 5 plots (PR, ROC, CM, cost)
│   └── pipeline.py                   # End-to-end configurable pipeline
│
├── scripts/
│   ├── train.py                       # Train single or all models
│   └── predict.py                     # Score transactions + risk tier assignment
│
├── tests/                             # 75 unit tests
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_features.py
│   ├── test_trainers.py
│   └── test_evaluation.py
│
├── configs/config.yaml                # Centralized hyperparameters
├── experiments/                       # MLflow SQLite (gitignored)
├── outputs/
│   ├── models/                        # Trained .joblib artifacts
│   └── reports/                       # Metrics + high-risk transaction lists
├── Makefile
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
pip install -e ".[dev,notebooks]"
```

### 2. Data Scientist Workflow
```bash
make notebooks   # Opens all 6 notebooks in Jupyter Lab
```

| Notebook | Focus |
|----------|-------|
| `01_EDA.ipynb` | Understand imbalance, temporal & geographic fraud patterns |
| `02_Feature_Engineering.ipynb` | Build & validate 10 engineered features |
| `03_Imbalance_Strategies.ipynb` | ⭐ Compare sampling strategies (unique notebook) |
| `04_Model_Experiments.ipynb` | Train 4 models, compare PR-AUC & F2 |
| `05_Model_Interpretation.ipynb` | SHAP explanations per fraud transaction |
| `06_Business_Report.ipynb` | Cost-benefit analysis & risk tier segmentation |

### 3. ML Engineer Workflow
```bash
make smoke              # Quick sanity check (10K rows)
make train              # Train HistGradientBoosting (default)
make train-all          # Train all 4 models
make predict            # Score test set + risk tiers
make mlflow-ui          # → http://localhost:5000
```

### 4. Run Tests
```bash
make test   # 75 tests with coverage
```

---

## Why This Dataset is Challenging

| Challenge | Impact | Our Solution |
|-----------|--------|--------------|
| **1:175 class imbalance** | Accuracy useless | PR-AUC as primary metric |
| **Threshold irrelevant at 0.5** | Misses all fraud | F2 + cost-optimized threshold |
| **FP/FN cost asymmetry** | Miss fraud = 50× worse | Cost matrix optimization |
| **Geospatial features** | Raw coords uninformative | Haversine distance engineering |
| **Heavy amount skew** | Model bias | log1p transform |

---

## Features (10 Engineered)

| Feature | Description | Fraud Signal |
|---------|-------------|--------------|
| `log_amt` | log1p(transaction amount) | High amount → higher fraud |
| `log_distance_km` | log1p(Haversine customer↔merchant) | ⭐ Strongest — fraud is geographically far |
| `log_city_pop` | log1p(city population) | — |
| `amt_to_pop_ratio` | High spend in small city | Suspicious ratio |
| `age` | Customer age at transaction | — |
| `hour` | Hour of day (0-23) | Night hours elevated |
| `day_of_week` | 0=Mon…6=Sun | Weekend patterns |
| `is_night` | 1 if 11pm–5am | ⭐ Elevated fraud rate |
| `is_weekend` | 1 if Sat/Sun | Modest effect |
| `category` | Merchant category (OHE) | ⭐ Strong predictor |

---

## Models & Imbalance Handling

| Algorithm | Imbalance Strategy | Primary Tuning |
|-----------|-------------------|----------------|
| **HistGradientBoosting** | `class_weight='balanced'` | learning_rate, max_depth |
| **XGBoost** | `scale_pos_weight` (auto-computed) | subsample, colsample |
| **LightGBM** | `is_unbalance=True` | n_estimators, max_depth |
| **Logistic Regression** | `class_weight='balanced'` | C, solver |

---

## Metrics

| Metric | Rationale |
|--------|-----------|
| **PR-AUC** | Primary — handles severe imbalance; ROC-AUC is optimistic |
| **F2-Score** | β=2: recall weighted 2× (missing fraud > false alarm) |
| **Fraud Recall** | % of actual fraud caught — business critical |
| **Fraud Precision** | % of fraud alerts that are real — operational cost |
| **Cost ($)** | Total: FP×$10 + FN×$500 — financial impact |

---

## MLflow Tracking

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlruns.db --port 5000
```

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make notebooks` | Open 6 notebooks in Jupyter Lab |
| `make smoke` | Quick 10K-row training check |
| `make train` | Train default model |
| `make train-all` | Train all 4 models |
| `make predict` | Score test set |
| `make test` | Run test suite |
| `make mlflow-ui` | Launch experiment tracker |
