from fastapi import FastAPI
import pandas as pd
import joblib
import os

app = FastAPI(title="Hospital Risk Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_healthcare_risk_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

# Load model and features
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # ensure correct feature order
    df = df.reindex(columns=feature_names, fill_value=0)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "risk": "HIGH" if int(pred) == 1 else "LOW",
        "probability": round(float(prob), 4)
    }

