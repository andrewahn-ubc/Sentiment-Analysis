import random
from transformers import pipeline

class ModelRegistry():
    def __init__(self):
        self.models = {
            "distilbert": self.model_a_predict,
            "roberta": self.model_b_predict
        }
        # For A/B testing ratio
        self.weights = {
            "distilbert": 0.5,
            "roberta": 0.5
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