# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("credit_api")

# Create input/output pydantic models
input_model = create_model("credit_api_input", **{'LIMIT_BAL': 170000.0, 'SEX': 1.0, 'EDUCATION': 1.0, 'MARRIAGE': 1.0, 'AGE': 41.0, 'PAY_1': 0.0, 'PAY_2': 0.0, 'PAY_3': 0.0, 'PAY_4': 0.0, 'PAY_5': 0.0, 'PAY_6': 0.0, 'BILL_AMT1': 149941.0, 'BILL_AMT2': 68912.0, 'BILL_AMT3': 72741.0, 'BILL_AMT4': 76149.0, 'BILL_AMT5': 84474.0, 'BILL_AMT6': 92400.0, 'PAY_AMT1': 3200.0, 'PAY_AMT2': 6000.0, 'PAY_AMT3': 5000.0, 'PAY_AMT4': 10000.0, 'PAY_AMT5': 10000.0, 'PAY_AMT6': 780.0, 'avg_bill': 90769.5, 'avg_payment': 5830.0, 'utilization_rate': 0.5339382290840149, 'payment_ratio': 0.06422862410545349, 'total_delay_months': 0.0, 'max_delay': 0.0})
output_model = create_model("credit_api_output", prediction=0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
