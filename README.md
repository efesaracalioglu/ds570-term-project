# F1 Pit Stop Strategy Predictor

An end-to-end data science pipeline that predicts **whether a driver will pit on the next lap**, using race telemetry and tyre data from F1 seasons 2022–2025. Results are served via an interactive Streamlit dashboard.

## Problem

Pit stop timing is one of the highest-leverage decisions in an F1 race. A one-lap error can cost positions or even a race win. Given the current lap's tyre age, lap time trend, race progress, and compound type, this project predicts the binary outcome: **pit next lap (1) / stay out (0)**.

## Data

**Source:** [F1 Strategy Dataset — Pit Stop Prediction](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/data) (Kaggle, public)  
101,305 laps · 4 seasons (2022–2025) · 16 features · target: `PitNextLap`  
Class distribution: 74.5% no-pit / 25.5% pit (2.93:1 ratio — mild imbalance)

## Methods

Five models trained and compared to justify algorithm selection:

| Model | Role |
|---|---|
| Decision Tree | Simplest tree baseline — interpretable, prone to overfit |
| Logistic Regression | Linear baseline — well-calibrated probabilities |
| Random Forest | Ensemble baseline — reduces DT variance without boosting |
| MLP | Neural network baseline — non-linear, no temporal structure |
| **XGBoost** | **Primary model** — gradient boosting, best overall performance |

A naive benchmark (always predict "no pit") scores 74.5% accuracy but catches zero pit laps — this is why accuracy is excluded as a primary metric.

**Validation:** Temporal split (first 80% → train, last 20% → test). No shuffling — preserves chronological order to prevent data leakage. Cross-validation uses TimeSeriesSplit (5 folds) so future laps never appear in training folds.  
**Imbalance handling:** `scale_pos_weight` (XGBoost), `class_weight='balanced'` (others).  
**Primary metrics:** ROC-AUC and F1. Precision and Recall also reported.

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Decision Tree | 0.808 | 0.661 | 0.533 | 0.869 |
| Logistic Regression | 0.751 | 0.579 | 0.462 | 0.775 |
| Random Forest | 0.845 | 0.682 | 0.562 | 0.867 |
| MLP | 0.798 | 0.562 | 0.636 | 0.503 |
| **XGBoost** | **0.879** | **0.720** | **0.652** | **0.804** |

Top features by XGBoost importance: `TyreLife`, `tyre_age_sq`, `LapNumber`, `Normalized_TyreLife`.

## Limitations

- Season-to-season regulation changes may hurt generalisation across years.
- Driver identity and team strategy are not encoded — same telemetry can mean different pit decisions depending on the constructor.
- Safety car / VSC periods are not explicitly flagged, causing opportunistic pit stops to be harder to predict.
- Train (23.8%) and test (32.3%) pit rates differ due to race mix in the temporal split — a form of distribution shift.

## Stack

Python 3.11 · XGBoost · scikit-learn · pandas · Streamlit · Plotly · Docker

## Project Structure

```
├── data/
│   └── f1_strategy_dataset_v4.csv
├── src/
│   ├── data_loader.py    # CSV loading
│   ├── preprocessing.py  # cleaning, column standardisation
│   ├── features.py       # compound encoding, tyre_age_sq
│   ├── model.py          # 5 models, temporal split, TimeSeriesSplit CV
│   └── evaluate.py       # ROC-AUC, F1, Precision, Recall, PR curve
├── app/
│   └── dashboard.py      # Streamlit — 4 interactive tabs
├── tests/
│   └── test_preprocessing.py
├── main.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## How to Run

### Docker (recommended)

```bash
docker compose up
```

Open **http://localhost:8501** in your browser. The pipeline runs automatically inside the container — no manual steps needed.

### Local

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## Dashboard

- **Data Explorer** — raw data summary, KPIs, pit rates by compound/season/tyre age, class imbalance chart, correlation heatmap
- **Model Comparison** — ROC and Precision-Recall curves for all 5 models, metrics table with cross-validation scores
- **XGBoost Details** — confusion matrix, feature importance, classification report, model limitations
- **Race Strategy** — select any race + driver → predicted pit-stop probability per lap with actual pit markers

## Course

DS 570 — Practical Applications of Data Science · Final Project
