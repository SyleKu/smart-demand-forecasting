# Smart Demand Forecasting

An end-to-end machine learning system for energy load forecasting, covering data preprocessing, 
feature engineering, model training, evaluation, API deployment, and interactive visualization.

---

## Results

### Prediction vs Actual
![Prediction vs Actual](reports/figures/prediction_vs_actual.png)

### Residual Distribution
![Prediction vs Actual](reports/figures/residuals.png)

### Absolute Error Over Time
![Prediction vs Actual](reports/figures/absolute_error_over_time.png)

---

## Dashboard
![Streamlit Dashboard](assets/dashboard_example.png)

The project includes an interactive Streamlit dashboard for exploring model predictions and evaluation results.

### Features
- Manual input of engineered forecasting features
- Real-time model inference
- Display of evaluation metrics
- Visualization of forecasting results and residual analysis

### Run dashboard
```
streamlit run dashboard/app.py
```

---

## Feature Engineering

The model uses a combination of:

- Time-based features (hours, day of week, month)
- Lag features to capture temporal dependencies
- Rolling statistics to capture short-term trends and variability

This allows the model to learn both periodic patterns and recent dynamics in the data.

---

## Motivation

This project was designed to demonstrate applied machine learning and data science workflow development in a complete, reproducible system.
The focus is not only on model performance, but also on engineering structure, interpretability, and deployment. 

---

## Tech Stack

- Python
- pandas / numpy
- scikit-learn
- XGBoost
- FastAPI
- Streamlit
- matplotlib / plotly

---

```
smart-demand-forecasting/
├── app/
├── dashboard/
├── data/
│   └── processed/
│   └── raw/
│   └── sample/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```
