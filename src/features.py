import pandas as pd

COMPOUND_ORDER = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

FEATURE_COLS = [
    "LapNumber",
    "TyreLife",
    "LapTimeSec",
    "Position",
    "Stint",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Normalized_TyreLife",
    "Position_Change",
    "Year",
    "compound_enc",
    "tyre_age_sq",
]


def engineer(df: pd.DataFrame, col: dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    df["tyre_age_sq"] = df["TyreLife"] ** 2

    compound_upper = df["Compound"].str.upper().str.strip()
    df["compound_enc"] = compound_upper.map(COMPOUND_ORDER).fillna(-1).astype(int)

    print(f"[features] Added: tyre_age_sq, compound_enc")
    return df


def get_feature_cols() -> list[str]:
    return FEATURE_COLS
