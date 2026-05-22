
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURE_COLS


def temporal_split(df: pd.DataFrame, target_col: str, test_frac: float = 0.2):
    df_model = df[FEATURE_COLS + [target_col]].dropna()
    split = int(len(df_model) * (1 - test_frac))
    X = df_model[FEATURE_COLS]
    y = df_model[target_col].astype(int)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def build_baseline(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def build_xgb(X_train, y_train) -> XGBClassifier:
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train(df: pd.DataFrame, col: dict[str, str]):
    target_col = col["target"]
    X_train, X_test, y_train, y_test = temporal_split(df, target_col)

    print(f"[model] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"[model] Train pit rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    baseline = build_baseline(X_train, y_train)
    print("[model] Baseline (Logistic Regression) trained")

    xgb = build_xgb(X_train, y_train)
    print("[model] XGBoost trained")

    return xgb, baseline, X_train, X_test, y_train, y_test
