#!/usr/bin/env python3
"""
Setup script to download Real-ESRGAN models to Modal persistent storage.
Run this once after deployment to populate the model cache.
"""

import modal

def setup_models():
    """Download models to persistent storage"""
    print("🚀 Setting up Real-ESRGAN models in persistent storage...")
    
    # Import the app and download function
    from main import app, download_models
    
    # Run the download function
    with app.run():
        print("📥 Downloading models...")
        download_models.remote()
        print("✅ Models setup complete!")
        print("🎯 Your upscaler is now optimized for fast warm starts!")

if __name__ == "__main__":
    setup_models()