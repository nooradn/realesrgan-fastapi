#!/usr/bin/env python3
"""
Setup script to download Real-ESRGAN models to Modal persistent storage.
Run this once after deployment to populate the model cache.
"""

def main():
    """Download models to persistent storage"""
    print("🚀 Setting up Real-ESRGAN models in persistent storage...")
    
    # Import the app and setup function
    from core.config import app
    from core.upscaler import setup_models
    
    # Run the setup function
    with app.run():
        print("📥 Downloading models...")
        result = setup_models.remote()
        print(f"✅ {result}")
        print("🎯 Your upscaler is now optimized for fast warm starts!")

if __name__ == "__main__":
    main()