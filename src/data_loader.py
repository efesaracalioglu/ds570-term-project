from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/f1_strategy_dataset_v4.csv")


def load() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH)
    print(f"[data_loader] Loaded {len(df):,} rows, {df.shape[1]} columns from {DATA_PATH.name}")
    return df
