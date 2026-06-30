from pydantic import BaseModel, Field
from typing import Literal, List

""" [

    "age",
    "sex",
    "chest_pains",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "rest_ecg",
    "max_heart_rate",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "num_major_vessels",
    "thalassemia",
    "target",
]
"""
class HeartDisease(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Age of the patient")
    sex: Literal['male', 'female'] = Field(..., description="Gender")
    chest_pains: Literal[1, 2, 3, 4] = Field(
        ..., description="1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic"
    )
    resting_blood_pressure: int = Field(..., ge=80, le=220, description="Resting blood pressure (mm Hg)")
    cholesterol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fasting_blood_sugar: bool = Field(..., description="True if fasting blood sugar > 120 mg/dl")
    rest_ecg: Literal[0, 1, 2] = Field(
        ..., description="0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy"
    )
    max_heart_rate: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exercise_induced_angina: bool = Field(..., description="True if exercise induced angina")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: Literal[1, 2, 3] = Field(
        ..., description="1=upsloping, 2=flat, 3=downsloping"
    )
    thalassemia: Literal[3, 6, 7] = Field(
        ..., description="3=normal, 6=fixed defect, 7=reversable defect"
    )


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(..., description="0=no heart disease, 1=heart disease probably present please consult a doctor")
