"""
Voice listing endpoint
"""

from fastapi import APIRouter
from app.models import VoicesResponse, VoiceInfo
from app.config import Config

router = APIRouter()

# Voice descriptions for built-in speakers
VOICE_DESCRIPTIONS = {
    "Vivian": "Female voice, clear and expressive",
    "Serena": "Female voice, warm and soothing",
    "Uncle_Fu": "Male voice, mature and authoritative",
    "Dylan": "Male voice, young and energetic",
    "Eric": "Male voice, professional and clear",
    "Ryan": "Male voice, friendly and casual",
    "Aiden": "Male voice, youthful and dynamic",
    "Ono_Anna": "Female voice, Japanese accent, gentle",
    "Sohee": "Female voice, Korean accent, pleasant",
}


@router.get("/voices", response_model=VoicesResponse)
async def list_voices():
    """List available voices"""
    voices = []
    for speaker in Config.SPEAKERS:
        voices.append(VoiceInfo(
            voice_id=speaker.lower(),
            name=speaker,
            language="multilingual",
            description=VOICE_DESCRIPTIONS.get(speaker, "Built-in speaker")
        ))

    return VoicesResponse(voices=voices)


@router.get("/audio/voices", response_model=VoicesResponse)
async def list_audio_voices():
    """List available voices (OpenAI-compatible path)"""
    return await list_voices()
