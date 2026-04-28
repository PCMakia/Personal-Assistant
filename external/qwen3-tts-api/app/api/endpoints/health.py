"""
Health check endpoints
"""

from fastapi import APIRouter
from app.models import HealthResponse
from app.config import Config
from app.tts_model import is_model_loaded, is_voice_clone_loaded, is_voice_design_loaded

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok" if is_model_loaded() else "initializing",
        model_loaded=is_model_loaded(),
        voice_clone_loaded=is_voice_clone_loaded(),
        voice_design_loaded=is_voice_design_loaded(),
        device=Config.DEVICE,
        model_name=Config.MODEL_NAME
    )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Qwen3-TTS API",
        "version": "1.0.0",
        "status": "ok" if is_model_loaded() else "initializing"
    }
