from fastapi import FastAPI, HTTPException
import uvicorn
import random
from transformers import pipeline
import time
from collections import defaultdict
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Sentiment Analysis Platform")

def create_empty_metrics():
    return {"count": 0,
            "total_latency": 0,
            "errors": 0}

metrics = defaultdict(create_empty_metrics)

class ModelRegistry():
    def __init__(self):
        self.models = {
            "model_a": self.model_a_predict,
            "model_b": self.model_b_predict
        }
        # For A/B testing ratio
        self.weights = {
            "model_a": 0.5,
            "model_b": 0.5
        }       
        print("Loading distillbert")
        self.model_a = pipeline("sentiment-analysis")
        print("Loading roberta")
        self.model_b = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def model_a_predict(self, text: str) -> dict:
        result = self.model_a(text)[0]        # extracts the only result in the list of results, assuming 1 pred at a time
        return {
            "prediction": result["label"],
            "confidence": result["score"],
            "model_version": "distilbert"
        }


    def model_b_predict(self, text: str) -> dict:
        result = self.model_b(text)[0]        # extracts the only result in the list of results, assuming 1 pred at a time
        return {
            "prediction": result["label"],
            "confidence": result["score"],
            "model_version": "roberta"
        }

    def select_model(self) -> str:
        return random.choices(
            list(self.models.keys()),
            weights=list(self.weights.values())
        )[0]

    def predict(self, model_name: str, text: str) -> dict:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} is not found.")
        return self.models[model_name](text)


registry = ModelRegistry()

class PredictionRequest(BaseModel):
    text: str
    model: str = None       # if not specified, default to A/B testing

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str
    latency: float          # measured in ms
    timestamp: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Main prediction endpoint with A/B testing
    start_time = time.time()

    input_text = request.text
    model_name = request.model

    try:
        model_name = model_name if model_name else registry.select_model()

        result = registry.predict(model_name, input_text)

        metrics[model_name]["count"] += 1

        end_time = time.time()
        latency = (end_time - start_time) * 1000
        metrics[model_name]["total_latency"] += latency

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_version=result["model_version"],
            latency=round(latency,2),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        metrics[model_name]["errors"] += 1
        return HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
