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
# For larger images, consider upgrading to gpu="A10G" or gpu="A100"
@app.function(
    image=gpu_image,
    gpu="T4",  # Options: "T4" (14GB), "A10G" (24GB), "A100" (40GB)
    timeout=300,
    memory=8192,
    volumes={"/models": model_volume, "/temp": temp_volume}
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
        # Clear GPU cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        original_size = list(image.size)
        
        # Check image size and resize if too large
        max_dimension = 2048  # Limit max dimension to prevent OOM
        if max(original_size) > max_dimension:
            ratio = max_dimension / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"⚠️ Image resized from {original_size} to {new_size} to prevent OOM")
        
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
        
        # Memory optimization settings
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=512,  # Use tiling to reduce memory usage
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False,  # Use FP16 to save memory
            gpu_id=0 if torch.cuda.is_available() else None
        )
        
        # Upscale image
        output, _ = upsampler.enhance(img_array, outscale=scale)
        
        # Convert back to PIL Image
        if len(output.shape) == 3:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        result_image = Image.fromarray(output)
        upscaled_size = list(result_image.size)
        
        # Save to temporary storage with timestamp
        import uuid
        import time
        from datetime import datetime, timedelta
        
        file_id = str(uuid.uuid4())
        timestamp = int(time.time())
        filename = f"{file_id}_{timestamp}.png"
        temp_path = f"/temp/{filename}"
        
        # Save image to temporary volume
        result_image.save(temp_path, format='PNG')
        temp_volume.commit()
        
        # Calculate expiry time (1 hour from now)
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "file_id": file_id,
            "filename": filename,
            "original_size": original_size,
            "upscaled_size": upscaled_size,
            "expires_at": expires_at.isoformat() + "Z"
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
        download_url: str  # URL to download result
        file_id: str  # Unique file identifier
        original_size: list
        upscaled_size: list
        expires_at: str  # ISO timestamp when file expires
    
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
    async def upscale_endpoint(
        upscale_request: UpscaleRequest, 
        request: Request,
        token: str = Depends(verify_token)
    ):
        try:
            # Validate scale parameter
            if upscale_request.scale not in [2, 4]:
                raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
            
            # Validate base64 image
            if not upscale_request.image:
                raise HTTPException(status_code=400, detail="Image data is required")
            
            # Call the GPU function for processing
            result = process_upscale.remote(upscale_request.image, upscale_request.scale)
            
            # Generate download URL using request headers
            base_url = f"{request.url.scheme}://{request.url.netloc}"
            download_url = f"{base_url}/download/{result['file_id']}"
            
            return UpscaleResponse(
                download_url=download_url,
                file_id=result["file_id"],
                original_size=result["original_size"],
                upscaled_size=result["upscaled_size"],
                expires_at=result["expires_at"]
            )
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    # GET to /upscale should return helpful message instead of crashing
    @web_app.get("/download/{file_id}")
    async def download_file(file_id: str):
        """Download upscaled image file"""
        import os
        import glob
        from fastapi.responses import FileResponse
        
        try:
            # Find file with this ID (includes timestamp in filename)
            pattern = f"/temp/{file_id}_*.png"
            matching_files = glob.glob(pattern)
            
            if not matching_files:
                raise HTTPException(status_code=404, detail="File not found or expired")
            
            file_path = matching_files[0]
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found or expired")
            
            # Return file for download
            return FileResponse(
                path=file_path,
                filename=f"upscaled_{file_id}.png",
                media_type="image/png"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")
    
    @web_app.get("/upscale")
    async def upscale_get_info():
        return {
            "error": "Method not allowed",
            "message": "Use POST method to upscale images",
            "required_fields": {
                "image": "base64 encoded image string",
                "scale": "2 or 4"
            },
            "response_format": {
                "download_url": "URL to download result image",
                "expires_at": "File expiry time (1 hour)"
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

# Auto cleanup function - runs every hour to delete expired files
@app.function(
    image=web_image,
    schedule=modal.Cron("0 * * * *"),  # Every hour at minute 0
    volumes={"/temp": temp_volume}
)
def cleanup_expired_files():
    """Delete files older than 1 hour"""
    import os
    import time
    import glob
    
    current_time = int(time.time())
    deleted_count = 0
    
    # Find all PNG files in temp directory
    temp_files = glob.glob("/temp/*.png")
    
    for file_path in temp_files:
        try:
            # Extract timestamp from filename: {uuid}_{timestamp}.png
            filename = os.path.basename(file_path)
            if "_" in filename:
                timestamp_str = filename.split("_")[1].split(".")[0]
                file_timestamp = int(timestamp_str)
                
                # Check if file is older than 1 hour (3600 seconds)
                if current_time - file_timestamp > 3600:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"🗑️ Deleted expired file: {filename}")
                    
        except (ValueError, IndexError, OSError) as e:
            print(f"⚠️ Error processing file {file_path}: {e}")
            continue
    
    # Commit changes to volume
    if deleted_count > 0:
        temp_volume.commit()
        print(f"✅ Cleanup complete: {deleted_count} expired files deleted")
    else:
        print("✅ Cleanup complete: No expired files found")
    
    return f"Deleted {deleted_count} expired files"

