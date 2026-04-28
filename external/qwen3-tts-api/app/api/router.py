"""
API router - combines all endpoint routers
"""

from fastapi import APIRouter
from app.api.endpoints import speech, voices, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(speech.router, prefix="/v1", tags=["speech"])
api_router.include_router(voices.router, prefix="/v1", tags=["voices"])
