import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean, COL, TARGET
from src.features import engineer, get_feature_cols, FEATURE_COLS


def _make_df(n=100):
    rng = np.random.default_rng(42)
    laps = np.tile(np.arange(1, 51), n // 50 + 1)[:n]
    return pd.DataFrame({
        "Driver":                  np.repeat(["VER", "HAM"], n // 2),
        "LapNumber":               laps,
        "Compound":                rng.choice(["SOFT", "MEDIUM", "HARD"], n),
        "Stint":                   rng.integers(1, 4, n),
        "TyreLife":                rng.uniform(1, 40, n),
        "Position":                rng.integers(1, 20, n),
        "LapTime (s)":             rng.uniform(85, 95, n),
        "Race":                    "Bahrain Grand Prix",
        "Year":                    2023,
        "LapTime_Delta":           rng.uniform(-2, 2, n),
        "Cumulative_Degradation":  rng.uniform(-10, 0, n),
        "PitStop":                 rng.choice([0, 1], n, p=[0.93, 0.07]),
        "PitNextLap":              rng.choice([0, 1], n, p=[0.93, 0.07]),
        "RaceProgress":            np.linspace(0, 1, n),
        "Normalized_TyreLife":     rng.uniform(0, 1, n),
        "Position_Change":         rng.uniform(-3, 3, n),
    })


def test_clean_renames_laptime():
    df = _make_df()
    cleaned, _ = clean(df)
    assert "LapTimeSec" in cleaned.columns
    assert "LapTime (s)" not in cleaned.columns


def test_clean_target_is_pitnextlap():
    df = _make_df()
    _, col = clean(df)
    assert col["target"] == TARGET == "PitNextLap"


def test_clean_drops_null_compound():
    df = _make_df()
    df.loc[0, "Compound"] = np.nan
    cleaned, _ = clean(df)
    assert cleaned["Compound"].isna().sum() == 0


def test_clean_target_binary():
    df = _make_df()
    cleaned, col = clean(df)
    assert set(cleaned[col["target"]].unique()).issubset({0, 1})


def test_engineer_adds_new_cols():
    df = _make_df()
    df_clean, col = clean(df)
    df_eng = engineer(df_clean, col)
    assert "tyre_age_sq" in df_eng.columns
    assert "compound_enc" in df_eng.columns


def test_compound_enc_values():
    df = _make_df()
    df_clean, col = clean(df)
    df_eng = engineer(df_clean, col)
    
    assert set(df_eng["compound_enc"].unique()).issubset({0, 1, 2, 3, 4})


def test_no_data_leakage_temporal_split():
    """train_test_split(shuffle=False) must keep train before test (no temporal leakage)."""
    from src.model import temporal_split
    df = _make_df(200)
    df_clean, col = clean(df)
    df_eng = engineer(df_clean, col)
    X_tr, X_te, _, _ = temporal_split(df_eng, col["target"])
    
    assert X_tr.index.max() < X_te.index.min()


def test_target_not_in_features():
    """PitNextLap (target) must not appear in FEATURE_COLS."""
    assert "PitNextLap" not in FEATURE_COLS
    # PitStop (current lap) is a valid feature - different lap from the target
