import modal

# Modal setup
app = modal.App("image-upscaler-auth")

# Create persistent volume for model storage
model_volume = modal.Volume.from_name("realesrgan-models", create_if_missing=True)

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

# Setup models function (runs once to populate volume)
@app.function(
    image=gpu_image,
    volumes={"/models": model_volume},
    timeout=600
)
def setup_models():
    """Download Real-ESRGAN models to persistent storage - run this once for optimal performance"""
    import os
    import urllib.request
    
    model_dir = "/models"
    os.makedirs(model_dir, exist_ok=True)
    
    models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    }
    
    for model_name, url in models.items():
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"📥 Downloading {model_name}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Downloaded {model_name}")
        else:
            print(f"✅ {model_name} already exists")
    
    # Commit changes to volume
    model_volume.commit()
    print("🎯 All models ready in persistent storage!")
    return "Models setup complete"

# Real-ESRGAN processing function (GPU-enabled)
@app.function(
    image=gpu_image,
    gpu="T4",
    timeout=300,
    memory=8192,
    volumes={"/models": model_volume}
)
def process_upscale(image_base64: str, scale: int):
    """Process image upscaling with Real-ESRGAN on GPU using persistent models"""
    import io
    import base64
    import os
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        original_size = list(image.size)
        
        # Convert to numpy array
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Initialize Real-ESRGAN model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        
        # Model path with fallback logic
        if scale == 4:
            model_name = 'RealESRGAN_x4plus.pth'
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        else:
            model_name = 'RealESRGAN_x2plus.pth'
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        
        model_path = f'/models/{model_name}'
        
        # Check if model exists in persistent storage, if not download it
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found in persistent storage, downloading {model_name}...")
            os.makedirs('/models', exist_ok=True)
            
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            print(f"✅ Downloaded {model_name} to persistent storage")
            
            # Commit to volume for future use
            model_volume.commit()
        
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False,
            gpu_id=0 if torch.cuda.is_available() else None
        )
        
        # Upscale image
        output, _ = upsampler.enhance(img_array, outscale=scale)
        
        # Convert back to PIL Image
        if len(output.shape) == 3:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        result_image = Image.fromarray(output)
        upscaled_size = list(result_image.size)
        
        # Convert to base64
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "upscaled_image": result_base64,
            "original_size": original_size,
            "upscaled_size": upscaled_size
        }
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

# FastAPI web service (lightweight, no GPU)
@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name("upscaler-auth")]
)
@modal.asgi_app()
def fastapi_app():
    import os
    from fastapi import FastAPI, HTTPException, Request, Depends, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    
    # Get valid tokens from environment (set in Modal secrets)
    VALID_TOKENS = os.getenv("VALID_TOKENS", "your-secret-token-here,another-token").split(",")
    
    # Security
    security = HTTPBearer()
    
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token not in VALID_TOKENS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token
    
    # Request/Response models
    class UpscaleRequest(BaseModel):
        image: str  # base64 encoded image
        scale: int = 4  # upscale factor (2 or 4)

    class UpscaleResponse(BaseModel):
        upscaled_image: str  # base64 encoded result
        original_size: list
        upscaled_size: list
    
    # Create FastAPI app with error handling
    web_app = FastAPI(
        title="Image Upscaler API (Authenticated)", 
        version="1.0.0",
        description="Real-ESRGAN Image Upscaler with Bearer Token Authentication"
    )
    
    # Global exception handler to prevent crash loops
    @web_app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )
    
    # Method not allowed handler (prevents GET to POST endpoints)
    @web_app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc):
        return JSONResponse(
            status_code=405,
            content={
                "error": "Method not allowed",
                "detail": f"Method {request.method} not allowed for {request.url.path}",
                "allowed_methods": ["POST"] if request.url.path == "/upscale" else ["GET"]
            }
        )
    
    @web_app.post("/upscale", response_model=UpscaleResponse)
    async def upscale_endpoint(request: UpscaleRequest, token: str = Depends(verify_token)):
        try:
            # Validate scale parameter
            if request.scale not in [2, 4]:
                raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
            
            # Validate base64 image
            if not request.image:
                raise HTTPException(status_code=400, detail="Image data is required")
            
            # Call the GPU function for processing
            result = process_upscale.remote(request.image, request.scale)
            
            return UpscaleResponse(
                upscaled_image=result["upscaled_image"],
                original_size=result["original_size"],
                upscaled_size=result["upscaled_size"]
            )
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    # GET to /upscale should return helpful message instead of crashing
    @web_app.get("/upscale")
    async def upscale_get_info():
        return {
            "error": "Method not allowed",
            "message": "Use POST method to upscale images",
            "required_fields": {
                "image": "base64 encoded image string",
                "scale": "2 or 4"
            },
            "authentication": "Bearer token required in Authorization header"
        }
    
    @web_app.get("/health")
    def health():
        """Health check endpoint - no auth required"""
        return {"status": "healthy", "service": "image-upscaler-auth", "version": "1.0.0"}
    
    @web_app.get("/")
    def root():
        """Root endpoint with API info - no auth required"""
        return {
            "message": "Image Upscaler API with Authentication",
            "version": "1.0.0",
            "endpoints": {
                "POST /upscale": "Main upscaling endpoint (requires auth)",
                "GET /health": "Health check",
                "GET /docs": "API documentation",
                "GET /": "This info page"
            },
            "authentication": "Bearer token required for /upscale endpoint"
        }
    
    return web_app

