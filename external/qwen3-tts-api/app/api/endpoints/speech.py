"""
Speech synthesis endpoints - OpenAI compatible
"""

import io
import os
import tempfile
import asyncio
from typing import Optional

import soundfile as sf
from fastapi import APIRouter, HTTPException, status, Form, File, UploadFile
from fastapi.responses import StreamingResponse

from app.models import TTSRequest, VoiceCloneRequest, VoiceDesignRequest, ErrorResponse
from app.config import Config
from app.tts_model import (
    generate_custom_voice, generate_voice_clone, generate_voice_design,
    is_model_loaded, is_voice_clone_loaded, is_voice_design_loaded,
    initialize_voice_clone_model, initialize_voice_design_model
)

router = APIRouter()

SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}


def audio_to_bytes(audio_array, sample_rate: int, format: str = "wav") -> bytes:
    """Convert audio array to bytes in specified format"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": "No filename provided", "type": "invalid_request_error"}}
        )

    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported audio format: {file_ext}. Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
                    "type": "invalid_request_error"
                }
            }
        )


@router.post(
    "/audio/speech",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate speech from text",
    description="OpenAI-compatible text-to-speech endpoint. Uses built-in speakers."
)
async def text_to_speech(request: TTSRequest):
    """Generate speech from text using Qwen3-TTS"""

    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Model not loaded yet", "type": "model_error"}}
        )

    # Validate language
    language = request.language or Config.DEFAULT_LANGUAGE
    if language not in Config.LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported language: {language}. Supported: {', '.join(Config.LANGUAGES)}",
                    "type": "invalid_request_error"
                }
            }
        )

    # Map voice to speaker
    voice = request.voice or Config.DEFAULT_SPEAKER
    if voice not in Config.SPEAKERS:
        # Try case-insensitive match
        voice_lower = voice.lower()
        matched = next((s for s in Config.SPEAKERS if s.lower() == voice_lower), None)
        if matched:
            voice = matched
        else:
            print(f"Unknown voice '{voice}', using default '{Config.DEFAULT_SPEAKER}'")
            voice = Config.DEFAULT_SPEAKER

    try:
        loop = asyncio.get_event_loop()
        audio_array, sample_rate = await loop.run_in_executor(
            None,
            lambda: generate_custom_voice(
                text=request.input,
                language=language,
                speaker=voice,
                instruct=request.instruct
            )
        )

        # Convert to requested format
        format_map = {"wav": "wav", "mp3": "mp3"}
        output_format = format_map.get(request.response_format, "wav")
        audio_bytes = audio_to_bytes(audio_array, sample_rate, output_format)

        media_type = "audio/wav" if output_format == "wav" else "audio/mpeg"

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=speech.{output_format}"}
        )

    except Exception as e:
        print(f"TTS generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"TTS generation failed: {str(e)}", "type": "generation_error"}}
        )


@router.post(
    "/audio/speech/upload",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate speech with voice cloning",
    description="Clone a voice from an uploaded audio sample."
)
async def text_to_speech_with_upload(
    input: str = Form(..., description="Text to synthesize", min_length=1, max_length=5000),
    voice_file: UploadFile = File(..., description="Reference audio for voice cloning"),
    language: Optional[str] = Form("English", description="Language for synthesis"),
    ref_text: Optional[str] = Form(None, description="Transcript of reference audio (improves quality)"),
    x_vector_only: Optional[bool] = Form(False, description="Use speaker embedding only (faster)"),
    response_format: Optional[str] = Form("wav", description="Output format: wav, mp3")
):
    """Generate speech by cloning the uploaded voice"""

    # Lazy load voice clone model
    if not is_voice_clone_loaded():
        print("Loading voice clone model on first use...")
        await initialize_voice_clone_model()

    if not is_voice_clone_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Voice clone model failed to load", "type": "model_error"}}
        )

    # Validate audio file
    validate_audio_file(voice_file)

    # Validate language
    if language not in Config.LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported language: {language}. Supported: {', '.join(Config.LANGUAGES)}",
                    "type": "invalid_request_error"
                }
            }
        )

    temp_path = None
    try:
        # Save uploaded file to temp location
        file_ext = os.path.splitext(voice_file.filename.lower())[1]
        fd, temp_path = tempfile.mkstemp(suffix=file_ext, prefix="voice_ref_")
        file_content = await voice_file.read()
        with os.fdopen(fd, 'wb') as f:
            f.write(file_content)

        print(f"Voice clone: using {voice_file.filename} ({len(file_content):,} bytes)")

        loop = asyncio.get_event_loop()
        audio_array, sample_rate = await loop.run_in_executor(
            None,
            lambda: generate_voice_clone(
                text=input,
                ref_audio=temp_path,
                language=language,
                ref_text=ref_text,
                x_vector_only=x_vector_only or False
            )
        )

        # Convert to requested format
        format_map = {"wav": "wav", "mp3": "mp3"}
        output_format = format_map.get(response_format, "wav")
        audio_bytes = audio_to_bytes(audio_array, sample_rate, output_format)

        media_type = "audio/wav" if output_format == "wav" else "audio/mpeg"

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=speech.{output_format}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Voice clone failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Voice clone failed: {str(e)}", "type": "generation_error"}}
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@router.post(
    "/audio/speech/design",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate speech with designed voice",
    description="Create a voice from a natural language description and generate speech."
)
async def text_to_speech_voice_design(request: VoiceDesignRequest):
    """Generate speech with a voice designed from a text description"""

    # Lazy load voice design model
    if not is_voice_design_loaded():
        print("Loading voice design model on first use...")
        await initialize_voice_design_model()

    if not is_voice_design_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Voice design model failed to load", "type": "model_error"}}
        )

    # Validate language
    language = request.language or Config.DEFAULT_LANGUAGE
    if language not in Config.LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported language: {language}. Supported: {', '.join(Config.LANGUAGES)}",
                    "type": "invalid_request_error"
                }
            }
        )

    try:
        loop = asyncio.get_event_loop()
        audio_array, sample_rate = await loop.run_in_executor(
            None,
            lambda: generate_voice_design(
                text=request.input,
                language=language,
                voice_description=request.voice_description
            )
        )

        audio_bytes = audio_to_bytes(audio_array, sample_rate, "wav")

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )

    except Exception as e:
        print(f"Voice design generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": f"Voice design failed: {str(e)}", "type": "generation_error"}}
        )
