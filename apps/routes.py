from fastapi import APIRouter, HTTPException
import os
import joblib
import pandas as pd
import numpy as np

from apps.schemas import HeartDisease, PredictionResponse

from machine_learning.machine_learning import (
load_model,
predict_heart_disease
)

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "message": "API is healthy"}

@router.post("/predict", response_model=PredictionResponse)
def predict(disease: HeartDisease):
    try:
        print(f"DEBUG: Loading model...")
        model = load_model()["model"]
        print(f"DEBUG: MOdel loaded. Input: {disease.model_dump()}")
        data = disease.model_dump()
        data['sex'] = 1 if data['sex'] == 'male' else 0
        data['fasting_blood_sugar'] = int(data['fasting_blood_sugar'])
        data['exercise_induced_angina'] = int(data['exercise_induced_angina'])
        new_data = pd.DataFrame([data])
        print(f"DEBUG: DataFrame created: {new_data.columns.tolist()}")
        actual_prediction = predict_heart_disease(model, new_data)
        return PredictionResponse(prediction=int(actual_prediction[0]))
    except FileNotFoundError as e:
        print(f"DEBUG: FileNotFoundError: {e}")
        raise HTTPException(status_code=503, detail = "Model not found")
    except Exception as e:
        print(f"DEBUG: Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail = str(e))

@router.get("/model-info")
def model_info():
    path = os.getenv("MODEL_PATH", "models/heart_disease_model.joblib")

    if not os.path.exists(path):
        return {"status": "no model found"}

    model = joblib.load(path)['model']
    return{
        "model_type": type(model).__name__,
        "model_path": path,
    }
