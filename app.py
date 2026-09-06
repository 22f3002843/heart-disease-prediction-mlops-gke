import os
import json
import logging
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Set up root logger for Cloud Logging JSON output
logger = logging.getLogger("heart_disease_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

app = FastAPI(title="Heart Disease Prediction API")

MODEL_PATH = "model.joblib"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

class PatientData(BaseModel):
    age: float
    gender: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "heart-disease-gke"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        # Structured log payload for GCP Cloud Logging
        log_payload = {
            "severity": "INFO",
            "message": "Prediction executed",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "input_features": input_dict,
            "prediction": prediction,
            "prediction_label": "yes" if prediction == 1 else "no",
            "probability": round(probability, 4)
        }
        logger.info(json.dumps(log_payload))

        return {
            "prediction": prediction,
            "prediction_label": "yes" if prediction == 1 else "no",
            "probability": probability,
            "status": "success"
        }
    except Exception as e:
        error_payload = {
            "severity": "ERROR",
            "message": f"Prediction failed: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        logger.error(json.dumps(error_payload))
        raise HTTPException(status_code=500, detail=str(e))
