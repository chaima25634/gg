from pydantic import BaseModel
from typing import Optional

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class PredictionResponse(BaseModel):
    disease_key: str
    disease_name: str
    confidence: float
    severity: str
    symptoms: str
    cause: str
    treatment: list[str]
    prevention: list[str]
    leaf_detected: bool
    bounding_box: Optional[BoundingBox] = None
    all_probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = '1.0.0'
