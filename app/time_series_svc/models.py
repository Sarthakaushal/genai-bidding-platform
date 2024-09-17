from pydantic import BaseModel

class TimeSeriesRequest(BaseModel):
    series_id: int  # A list of time series data points
    target_feature: str  # The feature to be predicted (e.g., "value")
    context_len: int  # How many future points to generate