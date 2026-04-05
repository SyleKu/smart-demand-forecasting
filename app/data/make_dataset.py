import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / 'data/raw/energy.csv'
PROCESSED_DATA_PATH = BASE_DIR / 'data/processed/clean.csv'

def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort chronologically
    df = df.sort_values(["timestamp"])

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    return df

def save_processed_data(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main():
    print("Loading raw data...")
    df = load_raw_data(RAW_DATA_PATH)

    print("\nPreview_")
    print(df.head())

    print("\nCleaning data...")
    df = clean_data(df)

    print("\nStats:")
    print(df.describe())

    print(f"\nSaving cleaned data to {PROCESSED_DATA_PATH}...")
    save_processed_data(df, PROCESSED_DATA_PATH)

    print("\nDone!")

if __name__ == "__main__":
    main()
