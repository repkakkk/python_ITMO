import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("model_fitted.pkl", 'rb') as file:
    model = joblib.load(file)


class ModelRequestData(BaseModel):
    total_square: float
    rooms: int
    floor: int
    distance_to_center: float


class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


@app.post("/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.get("/predict_get")
def preprocess_data(total_square: float, rooms: int, floor: int, distance_to_center: float):
    data = ModelRequestData(total_square = total_square, rooms = rooms, floor = floor,  distance_to_center = distance_to_center)
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return JSONResponse(content={"message": result}, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug = True)
