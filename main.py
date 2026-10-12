from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(
    title="ModernBERT Subscription Detection API",
    version="1.0.0"
)

MODEL_PATH = Path.home() / "Downloads" / "modernbert_deploy"

tokenizer = None
model = None
device = torch.device("cpu")


@app.on_event("startup")
def load_model():
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )
    model.to(device)
    model.eval()


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "loading",
        "model": "ModernBERT"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())

    return {
        "prediction": model.config.id2label[predicted_id],
        "confidence": round(
            float(probabilities[predicted_id].item()), 4
        )
    }
