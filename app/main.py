from fastapi import FastAPI
from .routers import time_series_svc

app = FastAPI()

time_series_data = {}
columns = {}

# Include the router from the `items` and `users` submodules

app.include_router(time_series_svc.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI app!"}