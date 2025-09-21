"""
Configuration and Modal setup
"""
import modal

# Modal setup
app = modal.App("image-upscaler-auth")

# Create persistent volume for model storage
model_volume = modal.Volume.from_name("realesrgan-models", create_if_missing=True)

# Create volume for temporary result files (auto-cleanup)
temp_volume = modal.Volume.from_name("temp-results", create_if_missing=True)

# Lightweight image for FastAPI (no GPU dependencies)
web_image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi==0.104.1",
    "pydantic==2.5.0"
])

# Heavy image for Real-ESRGAN processing (with GPU dependencies)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        # OpenCV dependencies
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6", 
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libgcc-s1",
        "libstdc++6",
        # Additional system libs
        "wget",
        "curl",
        "ffmpeg",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "libgtk-3-dev",
        "libcanberra-gtk-module",
        "libcanberra-gtk3-module"
    ])
    .pip_install([
        "opencv-python-headless==4.8.1.78",
        "pillow==10.1.0",
        "numpy==1.24.3",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "basicsr==1.4.2",
        "realesrgan==0.3.0",
        "gfpgan==1.3.8"
    ])
)