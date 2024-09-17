from fastapi import APIRouter, HTTPException
from fastapi import File, UploadFile
import pandas as pd
from io import StringIO
from app.time_series_svc.models import TimeSeriesRequest

import pandas as pd
import matplotlib.pyplot as plt

from tsfm_public import (
    TinyTimeMixerForPrediction,
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.visualization import plot_predictions


# Instantiate the model.
zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
  "ibm-granite/granite-timeseries-ttm-v1", # Name of the model on HuggingFace.
  num_input_channels=1 # tsp.num_input_channels,
)

time_series_data = {}
columns = {}

router = APIRouter()




@router.get("/get_time_series")
async def read_time_series():
    return list(zip(time_series_data.keys(), [columns[key] for key in time_series_data.keys()]))

# api to upload time series data as csv to time series dictionary
@router.post("/time_series/upload")
async def upload_time_series(file: UploadFile = File(...)):
    series_id = len(time_series_data) + 1
    # Read the CSV file into a pandas DataFrame
    file_content = await file.read()
    # Convert the byte content to a StringIO object for pandas
    csv_data = StringIO(file_content.decode("utf-8"))
    # Read the CSV data into a pandas DataFrame
    df = pd.read_csv(csv_data)
    # Convert to datetime
    df['time'] = pd.to_datetime(df['time'])
    # df = pd.read_csv(file.file)
    # Convert the DataFrame to a list of dictionaries
    # data = df.to_dict(orient='records')
    # Store the data in the time series dictionary
    time_series_data[series_id] = df
    columns[series_id] = df.columns.tolist()
    
    return {"message": "Time series data uploaded successfully", "id":series_id}

# assuming data is in time series dictionary create a pipeline and predict the next context_len values
@router.post("/time_series/predict")
async def predict_time_series(request: TimeSeriesRequest):
    # load data from time series dictionary into a pandas dataframe
    df = pd.DataFrame(time_series_data[request.series_id])
    print(df.head())
    # Fill NA/NaN values by propagating the last valid value.
    input_df = df.ffill()

    # Only use the last `context_length` rows for prediction.
    input_df = input_df.iloc[-request.context_len:,]
    
    # Create a pipeline
    print(request.__dict__)
    pipeline = TimeSeriesForecastingPipeline(
        zeroshot_model,
        timestamp_column='time',
        id_columns=[],
        target_columns=request.target_feature,
        explode_forecasts=True,
        freq="h",
        device="cpu", # Specify your local GPU or CPU.
    )
    # Make a forecast on the target column given the input data.
    zeroshot_forecast = pipeline(input_df)
    zeroshot_forecast.tail()
    return zeroshot_forecast[-request.context_len:]


@router.post("/time_series/{series_id}/predict")
async def predict_time_series(series_id: str, request: TimeSeriesRequest):
    """
    Predict future time series values based on the input data, target feature,
    and generation length.

    Example of `data`:
    [
        {"timestamp": "2024-09-16T12:00:00Z", "value": 100, "temperature": 25},
        {"timestamp": "2024-09-16T12:05:00Z", "value": 105, "temperature": 24},
    ]

    Example of `target_feature`:
    "value"

    Example of `context_len`:
    10
    """
    if series_id not in time_series_data:
        time_series_data[series_id] = request.data
    else:
        time_series_data[series_id].extend(request.data)

    # Mock prediction logic for generating future points
    future_predictions = []
    last_data_point = time_series_data[series_id][-1]
    last_value = last_data_point.get(request.target_feature, None)

    if last_value is None:
        raise HTTPException(status_code=400, detail=f"Target feature {request.target_feature} not found")

    for i in range(request.generation_length):
        new_value = last_value + (i + 1) * 5  # Simple logic to increase the value
        future_predictions.append({
            "timestamp": f"future_{i + 1}",
            request.target_feature: new_value
        })

    return {
        "message": f"Predicted {request.generation_length} future points for {request.target_feature}",
        "predictions": future_predictions
    }