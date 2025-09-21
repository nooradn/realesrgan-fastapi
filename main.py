"""
Main entry point for Real-ESRGAN Image Upscaler
Imports all components and exposes the Modal app
"""

import modal

# Import all components to register them with the Modal app
from core.config import app
from core.upscaler import setup_models, process_upscale, get_file_content
from core.api import fastapi_app

# The app is now ready with all functions registered
# Deploy with: modal deploy main.py