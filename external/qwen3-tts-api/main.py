"""
Entry point for Qwen3-TTS API server

Environment variables:
  ENABLE_GRADIO=true  - Launch Gradio UI instead of API
  GRADIO_PORT=7860    - Port for Gradio UI (default: 7860)
"""

import os
import subprocess
import sys

import uvicorn
from app.config import Config

if __name__ == "__main__":
    enable_gradio = os.getenv("ENABLE_GRADIO", "false").lower() == "true"

    if enable_gradio:
        # Launch the official qwen-tts-demo Gradio UI
        gradio_port = os.getenv("GRADIO_PORT", "7860")
        flash_attn_flag = "--flash-attn" if Config.USE_FLASH_ATTENTION else "--no-flash-attn"

        cmd = [
            "qwen-tts-demo",
            Config.MODEL_NAME,
            "--device", Config.DEVICE,
            "--dtype", Config.DTYPE,
            flash_attn_flag,
            "--ip", Config.GRADIO_HOST,
            "--port", gradio_port,
        ]
        print(f"Launching Gradio UI: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        # Launch the FastAPI server
        uvicorn.run(
            "app.main:app",
            host=Config.HOST,
            port=Config.PORT,
            reload=False,
            workers=1  # Single worker for GPU model
        )
