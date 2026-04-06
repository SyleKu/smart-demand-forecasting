import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_PATH = BASE_DIR / "data/raw/energy.csv"

date_range = pd.date_range(start="2020-01-01", periods=500, freq="h")

np.random.seed(42)

data = pd.DataFrame({
    "timestamp": date_range,
    "demand": 100
    + 20 * np.sin(np.arange(len(date_range)) / 24 * 2 * np.pi)
    + np.random.normal(0, 5, len(date_range)),
})

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_PATH, index=False)

print(f"Dummy data saved to {OUTPUT_PATH}")
