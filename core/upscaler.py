"""
Real-ESRGAN image processing functions
"""
from .config import app, gpu_image, model_volume, temp_volume

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
    gpu="T4",  # Options: "T4" (14GB), "A10G" (24GB), "A100" (40GB)
    timeout=600,  # Increased timeout for batch processing
    memory=8192,
    max_containers=10,  # Limit concurrent GPU instances
    volumes={"/models": model_volume, "/temp": temp_volume}
)
def process_upscale(image_base64: str, scale: int, output_ext: str = "png"):
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
        import threading
        
        file_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # Determine file extension and format
        if output_ext.lower() == "jpg":
            file_ext = "jpg"
            save_format = "JPEG"
            # Convert RGBA to RGB for JPEG (no transparency support)
            if result_image.mode == "RGBA":
                result_image = result_image.convert("RGB")
        else:
            file_ext = "png"
            save_format = "PNG"
        
        filename = f"{file_id}_{timestamp}.{file_ext}"
        temp_path = f"/temp/{filename}"
        
        # Thread-safe file saving
        os.makedirs("/temp", exist_ok=True)
        
        # Save with appropriate quality for JPEG
        if save_format == "JPEG":
            result_image.save(temp_path, format=save_format, quality=97, optimize=True)
        else:
            result_image.save(temp_path, format=save_format)
        
        # Use thread lock for volume commit to avoid conflicts
        with threading.Lock():
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
            "expires_at": expires_at.isoformat() + "Z",
            "output_format": file_ext
        }
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

# Download function with volume access
@app.function(
    image=gpu_image,
    volumes={"/temp": temp_volume}
)
def get_file_content(file_id: str):
    """Get file content from temp storage"""
    import os
    import glob
    
    try:
        # Find file with this ID (includes timestamp and extension in filename)
        patterns = [f"/temp/{file_id}_*.png", f"/temp/{file_id}_*.jpg"]
        matching_files = []
        for pattern in patterns:
            matching_files.extend(glob.glob(pattern))
        
        if not matching_files:
            return {"error": f"File not found or expired. Pattern: {pattern}"}
        
        file_path = matching_files[0]
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File path does not exist: {file_path}"}
        
        # Read file content
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Get original extension from found file
        original_filename = os.path.basename(file_path)
        file_ext = original_filename.split('.')[-1]
        
        return {
            "content": file_content,
            "filename": f"upscaled_{file_id}.{file_ext}"
        }
        
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}