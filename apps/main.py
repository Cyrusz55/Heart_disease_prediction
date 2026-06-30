from fastapi import FastAPI
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from apps.routes import router

from scheduler import start_scheduler

from machine_learning.machine_learning import train_model

load_dotenv()

app = FastAPI(
    title = "Heart Disease Prediction API",
    description = "Predict presence or absence of heart disease",
    version = "1.0.0"
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # use specific origins for safety
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
def startup_event():
    from huggingface_hub import hf_hub_download
    os.makedirs("models", exist_ok=True)

    try:
        model_path = hf_hub_download(
            repo_id="cyrusnx/heart_disease_model",
            filename="heart_disease_model.joblib",
            cache_dir="models",
            token=os.getenv("HF_TOKEN")
        )
        print(f"Model downloaded from HuggingFace: {model_path}")
    except Exception as e:
        print(f"Could not download from HuggingFace: {e}")
        print("Attempting fallback: loading local model or training...")
        if not os.path.exists("models/heart_disease_model.joblib"):
            print("No Model found. Running initial training...")
            import pandas as pd
            from machine_learning.machine_learning import train_model
            try:
                df = pd.read_csv("clean_data/heart_disease_clened.csv")
                train_model(df)
            except FileNotFoundError:
                print("Warning: Training data not found. API will fail without model.")
    start_scheduler()
@app.get("/")
def root():
    html_path = "frontend/index.html"

    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Developer salary prediction model is running"}
