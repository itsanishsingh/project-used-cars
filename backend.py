from fastapi import FastAPI, Request
import joblib
import numpy as np
import pandas as pd

app = FastAPI()


def preprocessing(data):
    fields = [
        "normalized-losses",
        "drive-wheels",
        "engine-location",
        "wheel-base",
        "length",
        "width",
        "curb-weight",
        "engine-size",
        "fuel-system",
        "bore",
        "horsepower",
        "city-mpg",
        "highway-mpg",
    ]

    test_data = []

    for field in fields:
        test_data.append(data[field])

    test_data = np.array(test_data).reshape(1, 13)

    test_data = pd.DataFrame(test_data, columns=fields)

    pipe = joblib.load("preprocess_pipeline.joblib")

    return pipe.transform(test_data)


def prediction(data):
    test_data = preprocessing(data)

    model = joblib.load("model.joblib")

    result = model.predict(test_data)
    return result[0]


@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    result = prediction(data)
    response = {"Predicted price": result}
    return response
