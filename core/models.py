"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel
from typing import List

class UpscaleRequest(BaseModel):
    image: str  # base64 encoded image
    scale: int = 4  # upscale factor (2 or 4)

class UpscaleResponse(BaseModel):
    download_url: str  # URL to download result
    file_id: str  # Unique file identifier
    original_size: List[int]
    upscaled_size: List[int]
    expires_at: str  # ISO timestamp when file expires