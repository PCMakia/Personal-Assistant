"""
Main FastAPI application for Qwen3-TTS API
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Config
from app.tts_model import initialize_model
from app.api.router import api_router


ascii_art = r"""
  ___                 _____     _____ _____ ____
 / _ \__      _____ _|___ /    |_   _|_   _/ ___|
| | | \ \ /\ / / _ \ '_ \| |___  | |   | | \___ \
| |_| |\ V  V /  __/ | | |___ \ | |   | |  ___) |
 \__\_\ \_/\_/ \___|_| |_|____/ |_|   |_| |____/

"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print(ascii_art)
    print("Qwen3-TTS API Server")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Dtype: {Config.DTYPE}")
    print()

    # Start model loading in background
    model_task = asyncio.create_task(initialize_model())

    yield

    # Shutdown
    if not model_task.done():
        model_task.cancel()
        try:
            await model_task
        except asyncio.CancelledError:
            pass


# Create FastAPI app
app = FastAPI(
    title="Qwen3-TTS API",
    description="REST API for Qwen3-TTS with OpenAI-compatible endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
cors_origins = Config.CORS_ORIGINS
if cors_origins == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"error": {"message": str(exc.detail)}}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "internal_error"
            }
        }
    )
