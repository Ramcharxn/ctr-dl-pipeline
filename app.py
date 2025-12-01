from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from inference import predict_ctr_one

app = FastAPI()

@app.post("/predict")
def predict(req: Dict[str, Any]):
    # here you re-create the engineered feature row from raw fields
    raw_row = req
    # TODO: add your feature engineering: event_doc_age_hours, ctr features, etc.
    prob, label = predict_ctr_one(raw_row)
    return {"ctr_prob": prob, "clicked_pred": label}


@app.get("/health")
def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}