from pydantic import BaseModel

class ForecastRequest(BaseModel):
    hours: int
    day_of_week: int
    month: int
    is_weekend: bool
    lag_1: float
    lag_24: float
    rolling_mean_24: float
    rolling_std_24: float

class ForecastResponse(BaseModel):
    prediction: float
