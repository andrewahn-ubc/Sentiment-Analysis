from pydantic import BaseModel

class PredictionRequest(BaseModel):
    text: str
    model: str = None       # if not specified, default to A/B testing

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str
    latency: float          # measured in ms
    timestamp: str