import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_DATA_PATH = BASE_DIR / "data/processed/features.csv"
MODEL_PATH = BASE_DIR / "models/trained_model.pkl"
METRICS_PATH = BASE_DIR / "reports/metrics.json"
FIGURES_PATH = BASE_DIR / "reports/figures"

def load_data():
    df = pd.read_csv(FEATURE_DATA_PATH)
    model = joblib.load(MODEL_PATH)

    return df, model

def create_predictions(df, model):
    from app.core.config import FEATURE_COLUMNS, TARGET_COLUMN

    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    predictions = model.predict(X_test)

    return test_df, y_test, predictions

def plot_predictions(y_test, predictions):
    plt.figure()
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Predictions vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Demand")

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_PATH / "predictions_vs_actual.png")
    plt.close()

def plot_residuals(y_test, predictions):
    residuals = y_test.values - predictions

    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")

    plt.savefig(FIGURES_PATH / "residuals.png")
    plt.close()

def main():
    print("Loading data and model...")
    df, model = load_data()

    print("Generating predictions...")
    test_df, y_test, predictions = create_predictions(df, model)

    print("Plotting predictions vs actual...")
    plot_predictions(y_test, predictions)

    print("Plotting residuals...")
    plot_residuals(y_test, predictions)

    print("Done! Plots saved in reports/figures")

if __name__ == "__main__":
    main()
