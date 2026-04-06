import pandas as pd
from pathlib import Path
from app.core.config import FEATURE_COLUMNS, TARGET_COLUMN

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = BASE_DIR / 'data/processed/clean.csv'
FEATURE_DATA_PATH = BASE_DIR / 'data/processed/features.csv'

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Processed data not found at {path}')

    data = pd.read_csv(path)
    return data

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag_1"] = df[TARGET_COLUMN].shift(1)
    df["lag_24"] = df[TARGET_COLUMN].shift(24)
    return df

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rolling_mean_24"] = df[TARGET_COLUMN].rolling(window=24).mean()
    df["rolling_std_24"] = df[TARGET_COLUMN].rolling(window=24).std()
    return df

def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def validate_feature_columns(df: pd.DataFrame):
    expected_columns = ["timestamp", TARGET_COLUMN] + FEATURE_COLUMNS
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing feature columns after engineering: {missing_columns}")

def save_features(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main():
    print("Loading processed data...")
    df = load_data(PROCESSED_DATA_PATH)

    print("\nCreating time features...")
    df = create_time_features(df)

    print("\nCreating lag features...")
    df = create_lag_features(df)

    print("\nCreating rolling features...")
    df = create_rolling_features(df)

    print("\nColumns before cleaning:")
    print(df.columns.tolist())

    print("\nPreview of feature dataset:")
    print(df.head())

    print("\nCleaning feature dataset...")
    df = clean_feature_data(df)

    validate_feature_columns(df)

    print("\nFinal dataset preview:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nFeature statistics:")
    print(df.describe())

    print("\nSelected feature columns:")
    print(FEATURE_COLUMNS)

    print(f"\nSaving feature to { FEATURE_DATA_PATH }...")
    save_features(df, FEATURE_DATA_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()
