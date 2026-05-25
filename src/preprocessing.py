import pandas as pd

TARGET = "PitNextLap"

COL = {
    "lap":       "LapNumber",
    "driver":    "Driver",
    "compound":  "Compound",
    "tyre_age":  "TyreLife",
    "lap_time":  "LapTimeSec",
    "position":  "Position",
    "stint":     "Stint",
    "race":      "Race",
    "year":      "Year",
    "target":    TARGET,
    "lt_delta":         "LapTime_Delta",
    "cum_degradation":  "Cumulative_Degradation",
    "race_progress":    "RaceProgress",
    "norm_tyre_life":   "Normalized_TyreLife",
    "position_change":  "Position_Change",
}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    df = df.copy()

    # Rename column
    df = df.rename(columns={"LapTime (s)": "LapTimeSec"})

    df = df.dropna(subset=["Compound"])

    # We ensure target is integer
    df[TARGET] = df[TARGET].astype(int)

    # We sort for temporal integrity: Year → Race → Driver → Lap
    df = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).reset_index(drop=True)

    pit_rate = df[TARGET].mean()
    print(f"[preprocessing] {len(df):,} rows after cleaning | pit-next-lap rate: {pit_rate:.2%}")
    return df, COL
