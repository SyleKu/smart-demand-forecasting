import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
import streamlit as st

from app.core.config import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models/trained_model.pkl"
METRICS_PATH = BASE_DIR / "reports/metrics.json"
FIGURES_PATH = BASE_DIR / "reports/figures"

@st.cache_data
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

def build_input_dataframe(
        hour: int,
        day_of_week: int,
        month: int,
        is_weekend: bool,
        lag_1: float,
        lag_24: float,
        rolling_mean_24: float,
        rolling_std_24: float,
) -> pd.DataFrame:
    data = {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "rolling_mean_24": rolling_mean_24,
        "rolling_std_24": rolling_std_24,
    }

    df = pd.DataFrame([data])
    return df[FEATURE_COLUMNS]


def main():
    st.set_page_config(
        page_title="Smart Demand Forecasting",
        page_icon="📈",
        layout="wide",
    )

    st.title("Smart Demand Forecasting Dashboard")
    st.caption("This dashboard demonstrates the final stage of the end-to-end forecasting pipeline: "
               "interactive inference and result inspection.")
    st.markdown(
        "Interactive dashboard for energy load forecasting using a trained machine learning model.",
    )

    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    metrics = load_metrics()

    st.sidebar.header("Input Features")

    hour = st.sidebar.slider("Hour", min_value=0, max_value=23, value=14)
    day_of_week = st.sidebar.slider("Day Of Week", min_value=0, max_value=6, value=2)
    month = st.sidebar.slider("Month", min_value=1, max_value=12, value=1)
    is_weekend = st.sidebar.selectbox("Is Weekend", options=[0, 1], index=0)

    lag_1 = st.sidebar.number_input("Lag 1", value=120.5)
    lag_24 = st.sidebar.number_input("Lag 24", value=118.2)
    rolling_mean_24 = st.sidebar.number_input("Rolling Mean 24", value=119.1)
    rolling_std_24 = st.sidebar.number_input("Rolling Std 24", value=6.4)

    input_df = build_input_dataframe(
        hour=hour,
        day_of_week=day_of_week,
        month=month,
        is_weekend=is_weekend,
        lag_1=lag_1,
        lag_24=lag_24,
        rolling_mean_24=rolling_mean_24,
        rolling_std_24=rolling_std_24,
    )

    prediction = model.predict(input_df)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction")
        st.metric("Forecasted Demand", f"{prediction:.2f}")

        st.subheader("Input Sample")
        st.dataframe(input_df)

    with col2:
        st.subheader("Model Information")
        st.write("**Features used:**")
        st.write(FEATURE_COLUMNS)

        if metrics is not None:
            st.write("**Evaluation Metrics:**")

            baseline = metrics.get("baseline", {})
            random_forest = metrics.get("random_forest", {})

            metrics_df = pd.DataFrame(
                [
                    {
                        "Model": "Baseline",
                        "MAE": baseline.get("mae"),
                        "RMSE": baseline.get("rmse"),
                        "MAPE": baseline.get("mape"),
                    },
                    {
                        "Model": "Random Forest",
                        "MAE": random_forest.get("mae"),
                        "RMSE": random_forest.get("rmse"),
                        "MAPE": random_forest.get("mape"),
                    },
                ]
            )
            st.dataframe(metrics_df)
        else:
            st.info("No metrics.json found yet.")

    st.subheader("Evaluation Plots")

    prediction_plot = FIGURES_PATH / "prediction_vs_actual.png"
    residual_plot = FIGURES_PATH / "residuals.png"
    error_plot = FIGURES_PATH / "absolute_error_over_time.png"

    plot_col1, plot_col2, plot_col3 = st.columns(3)

    with plot_col1:
        if prediction_plot.exists():
            st.image(str(prediction_plot), caption="Prediction vs Actual")
        else:
            st.warning("prediction_vs_actual.png not found.")

    with plot_col2:
        if residual_plot.exists():
            st.image(str(residual_plot), caption="Residual Distribution")
        else:
            st.warning("residuals.png not found.")

    with plot_col3:
        if error_plot.exists():
            st.image(str(error_plot), caption="Absolute Error Over Time")
        else:
            st.warning("absolute_error_over_time.png not found.")


if __name__ == "__main__":
    main()



