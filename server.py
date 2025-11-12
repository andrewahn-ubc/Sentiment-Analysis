from fastapi import FastAPI, HTTPException
from collections import defaultdict
from datetime import datetime
from src.model_registry import ModelRegistry
from src.message_structure import PredictionRequest, PredictionResponse
import uvicorn
import time

app = FastAPI(title="Sentiment Analysis Platform")

registry = ModelRegistry()

def create_empty_metrics():
    return {"count": 0,
            "avg_latency_ms": 0,
            "total_latency": 0,
            "errors": 0,
            "error_rate": 0}
metrics = defaultdict(create_empty_metrics)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Main prediction endpoint with A/B testing
    start_time = time.time()

    input_text = request.text
    model_name = request.model

    if (model_name not in registry.models):
        raise HTTPException(status_code=400, detail="Requested model can't be found.")

    try:
        # Select model
        model_name = model_name if model_name else registry.select_model()

        # Predict
        result = registry.predict(model_name, input_text)

        # Update metrics
        metrics[model_name]["count"] += 1
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        metrics[model_name]["total_latency"] += latency
        metrics[model_name]["avg_latency_ms"] = round(metrics[model_name]["total_latency"] / metrics[model_name]["count"], 2)
        metrics[model_name]["error_rate"] = round(metrics[model_name]["errors"] / metrics[model_name]["count"], 4)

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
    
@app.post("/config/weights")
async def update_weights(weights: dict):
    # Ensure same number of models
    if (len(weights) != len(registry.weights)):
        raise HTTPException(status_code=400, detail="The input weights have a different number of models.")
    
    # Ensure the same model names
    for model in weights.keys():
        if model not in registry.weights:
            raise HTTPException(status_code=400, detail="At least one model is different.")
        
    # Ensure weights add up to 1
    if (sum(weights.values()) != 1):
        raise HTTPException(status_code=400, detail="Provided weights do not sum to 1.")

    # Update weights
    registry.weights = weights
    
    return {"message": "Weights successfully updated", "new_weights": weights}

@app.get("/config/weights")
def get_weights():
    return registry.weights
    
@app.get("/metrics")
def get_metrics():
    return metrics

@app.get("/health")
async def get_health():
    # For other APIs who want to ensure our API is running 
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
