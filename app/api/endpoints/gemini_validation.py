from fastapi import APIRouter
from google import genai
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/gemini", tags=["gemini"])


class GeminiValidationInput(BaseModel):
    api_key: str


class GeminiValidationResponse(BaseModel):
    valid: bool
    message: str


@router.post("/validation")
async def validate_gemini_api_key(data: GeminiValidationInput) -> GeminiValidationResponse:
    """Validate a Gemini API key by fetching a well-known model (free, no tokens used)."""
    try:
        client = genai.Client(api_key=data.api_key)
        await client.aio.models.list()
        return GeminiValidationResponse(valid=True, message="Gemini API key is valid")
    except Exception as e:
        logger.debug(f"Gemini API key validation failed: {e}")
        return GeminiValidationResponse(valid=False, message="Invalid Gemini API key")
