from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel

from model_service import OliveModelService
from image_processor import ImageProcessor
from schemas import PredictionResponse, HealthResponse

# ── RAG chatbot ────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_module'))
try:
    from chat_service import get_chat_response, get_prevention_alert
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    print(f"⚠️ RAG module not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_service = OliveModelService()
image_processor = ImageProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🫒 Loading Olive Disease Detection model...")
    await model_service.load_model()
    logger.info("✅ Model loaded successfully!")
    if RAG_AVAILABLE:
        logger.info("🤖 RAG chatbot available!")
    else:
        logger.warning("⚠️ RAG chatbot not available (Ollama not running?)")
    yield
    logger.info("🔴 Shutting down...")


app = FastAPI(
    title="Olive Leaf Disease Detection API",
    description="AI-powered olive leaf disease detection + RAG chatbot",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="online",
        model_loaded=model_service.is_loaded,
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="online" if model_service.is_loaded else "loading",
        model_loaded=model_service.is_loaded,
    )


# ── Image prediction ───────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB")

    try:
        processed_image, leaf_detected, bbox = await image_processor.detect_and_crop_leaf(image_bytes)
        prediction = await model_service.predict(processed_image)

        return PredictionResponse(
            disease_key=prediction["disease_key"],
            disease_name=prediction["disease_name"],
            confidence=prediction["confidence"],
            severity=prediction["severity"],
            symptoms=prediction["symptoms"],
            cause=prediction["cause"],
            treatment=prediction["treatment"],
            prevention=prediction["prevention"],
            leaf_detected=leaf_detected,
            bounding_box=bbox,
            all_probabilities=prediction["all_probabilities"],
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ── RAG Chatbot ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chatbot RAG multilingue.
    - Répond dans la langue de l'utilisateur (fr / en / darija)
    - Traduit la dernière réponse si l'utilisateur le demande
      (n'importe quelle formulation : "en français", "in english",
       "بالدارجة", "bel darija", "dis-moi en fr", etc.)
    """
    if not RAG_AVAILABLE:
        return {
            "response": "⚠️ Le chatbot n'est pas disponible. Vérifiez qu'Ollama est démarré.",
            "language": "fr",
            "translated": False
        }
    try:
        result = get_chat_response(request.message, request.session_id)
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Prevention alert ───────────────────────────────────────────────────────────

@app.get("/prevention")
async def prevention():
    """Alerte de prévention pour le mois actuel (3 langues)."""
    if not RAG_AVAILABLE:
        return {"error": "RAG module not available"}
    try:
        return get_prevention_alert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
