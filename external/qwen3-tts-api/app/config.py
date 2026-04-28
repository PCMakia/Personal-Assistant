"""
Configuration for Qwen3-TTS API
"""

import os


class Config:
    """Application configuration from environment variables"""

    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "4123"))
    GRADIO_HOST = os.getenv("GRADIO_HOST", HOST)

    # Model settings
    # CustomVoice model has 9 built-in speakers
    # Base model only supports voice cloning (no built-in speakers)
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    VOICE_CLONE_MODEL = os.getenv("VOICE_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    VOICE_DESIGN_MODEL = os.getenv("VOICE_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    DEVICE = os.getenv("DEVICE", "cuda:0")
    DTYPE = os.getenv("DTYPE", "bfloat16")  # bfloat16, float16, float32
    USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"

    # Model download settings
    # Set to true to prevent auto-downloading models (requires pre-downloaded weights)
    LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "true").lower() == "true"

    # Cache settings
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/cache")
    VOICE_LIBRARY_DIR = os.getenv("VOICE_LIBRARY_DIR", "/voices")

    # TTS settings
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "English")

    # Default speaker for custom voice generation
    DEFAULT_SPEAKER = os.getenv("DEFAULT_SPEAKER", "Ono_Anna")

    # Available speakers for custom voice
    SPEAKERS = [
        "Vivian", "Serena", "Uncle_Fu", "Dylan",
        "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"
    ]

    # Supported languages
    LANGUAGES = [
        "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese",
        "Spanish", "Italian"
    ]

    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
