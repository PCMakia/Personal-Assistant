"""
Pydantic models for API requests/responses
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request"""
    input: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    voice: Optional[str] = Field("Vivian", description="Voice name (speaker) or 'clone' for voice cloning")
    model: Optional[str] = Field(None, description="Model identifier (ignored, uses configured model)")
    response_format: Optional[str] = Field("wav", description="Audio format: wav, mp3")
    speed: Optional[float] = Field(1.0, description="Speed multiplier (not yet supported)")
    language: Optional[str] = Field("English", description="Language for synthesis")
    instruct: Optional[str] = Field(None, description="Instruction for how the voice should speak")


class VoiceCloneRequest(BaseModel):
    """Voice cloning request"""
    input: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    language: Optional[str] = Field("English", description="Language for synthesis")
    ref_text: Optional[str] = Field(None, description="Transcript of reference audio (improves quality)")
    x_vector_only: Optional[bool] = Field(False, description="Use speaker embedding only (faster, lower quality)")


class VoiceDesignRequest(BaseModel):
    """Voice design request - generate voice from description"""
    input: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    language: Optional[str] = Field("English", description="Language for synthesis")
    voice_description: str = Field(..., description="Natural language description of desired voice")


class VoiceInfo(BaseModel):
    """Voice information"""
    voice_id: str
    name: str
    language: Optional[str] = None
    description: Optional[str] = None


class VoicesResponse(BaseModel):
    """List of available voices"""
    voices: List[VoiceInfo]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    voice_clone_loaded: bool
    voice_design_loaded: bool
    device: str
    model_name: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: dict
