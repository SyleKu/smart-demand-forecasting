import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from app.core.config import FEATURE_COLUMNS, TARGET_COLUMN

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_DATA_PATH = BASE_DIR / "data/processed/features.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models/trained_model.pkl"
METRICS_OUTPUT_PATH = BASE_DIR / "reports/metrics.json"

def load_feature_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature data not found at {path}")

    df = pd.read_csv(path)

    print(f"\nColumns in dataset:")
    print(df.columns)

    print(f"\nDataset shape:")
    print(df.shape)

    print(f"\nPreview:")
    print(df.head())

    return df

def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df

def compute_baseline_predictions(test_df: pd.DataFrame) -> np.ndarray:
    # Simple baseline: use lag_1 as prediction
    return test_df["lag_1"].values

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(
            np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        ) * 100
    else:
        mape = None

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if mape is not None else None,
    }

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def main():
    print("Loading features data...")
    df = load_feature_data(FEATURE_DATA_PATH)

    print("\nSplitting data into train and test sets...")
    train_df, test_df = train_test_split_time_series(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nComputing baseline metrics...")
    baseline_preds = compute_baseline_predictions(test_df)
    baseline_metrics = compute_metrics(y_test.values, baseline_preds)

    print("Baseline metrics:")
    print(baseline_metrics)

    print("\nTraining RandomForest model...")
    model = train_model(X_train, y_train)

    print("Evaluating trained model...")
    model_preds = model.predict(X_test)
    model_metrics = compute_metrics(y_test.values, model_preds)

    print("Model metrics:")
    print(model_metrics)

    metrics = {
        "baseline": baseline_metrics,
        "random_forest": model_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }

    print(f"\nSaving model to {MODEL_OUTPUT_PATH}...")
    save_model(model, MODEL_OUTPUT_PATH)

    print(f"\nSaving metrics to {METRICS_OUTPUT_PATH}...")
    save_metrics(metrics, METRICS_OUTPUT_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()

