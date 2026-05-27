import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features import FEATURE_COLS

N_CV_SPLITS = 5


def temporal_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    """Split preserving temporal order via train_test_split(shuffle=False)."""
    df_model = df[FEATURE_COLS + [target_col]].dropna()
    X = df_model[FEATURE_COLS]
    y = df_model[target_col].astype(int)
    return train_test_split(X, y, test_size=test_size, shuffle=False)


def cv_score(model, X_train, y_train) -> dict:
    """
    Compute ROC-AUC via TimeSeriesSplit on the training set.
    Each fold's test window is strictly after its train window —
    no future data ever enters a training fold.
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    scores = cross_val_score(model, X_train, y_train,
                             cv=tscv, scoring="roc_auc", n_jobs=-1)
    return {"cv_mean": round(float(scores.mean()), 4),
            "cv_std":  round(float(scores.std()),  4)}


def build_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        min_samples_split=100,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def build_logistic_regression(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1_000,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def build_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def build_mlp(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )),
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
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=10,
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
    print(f"[model] CV strategy: TimeSeriesSplit (n_splits={N_CV_SPLITS})")

    dt = build_decision_tree(X_train, y_train)
    dt_cv = cv_score(dt, X_train, y_train)
    print(f"[model] Decision Tree     — CV AUC: {dt_cv['cv_mean']:.4f} ± {dt_cv['cv_std']:.4f}")

    lr = build_logistic_regression(X_train, y_train)
    lr_cv = cv_score(lr, X_train, y_train)
    print(f"[model] Logistic Reg.     — CV AUC: {lr_cv['cv_mean']:.4f} ± {lr_cv['cv_std']:.4f}")

    rf = build_random_forest(X_train, y_train)
    rf_cv = cv_score(rf, X_train, y_train)
    print(f"[model] Random Forest     — CV AUC: {rf_cv['cv_mean']:.4f} ± {rf_cv['cv_std']:.4f}")

    xgb = build_xgb(X_train, y_train)
    xgb_cv = cv_score(xgb, X_train, y_train)
    print(f"[model] XGBoost           — CV AUC: {xgb_cv['cv_mean']:.4f} ± {xgb_cv['cv_std']:.4f}")

    mlp = build_mlp(X_train, y_train)
    mlp_cv = cv_score(mlp, X_train, y_train)
    print(f"[model] MLP               — CV AUC: {mlp_cv['cv_mean']:.4f} ± {mlp_cv['cv_std']:.4f}")

    cv_results = {
        "Decision Tree":       dt_cv,
        "Logistic Regression": lr_cv,
        "Random Forest":       rf_cv,
        "XGBoost":             xgb_cv,
        "MLP":                 mlp_cv,
    }

    return xgb, lr, rf, dt, mlp, X_train, X_test, y_train, y_test, cv_results
