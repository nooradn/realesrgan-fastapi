# 🔒 Serverless Image Upscaler (Authenticated)

Real-ESRGAN image upscaler with FastAPI and Bearer token authentication, deployed on Modal.com.

## Features

- **🤖 Real-ESRGAN Model**: High-quality AI upscaling (2x, 4x)
- **🔒 Bearer Token Auth**: Secure API with authentication
- **⚡ Serverless Architecture**: 2-function design (FastAPI + GPU processing)
- **🚀 GPU Accelerated**: NVIDIA T4 for fast processing
- **🌐 Web Interface**: HTML interface for testing
- **📱 REST API**: FastAPI with automatic docs
- **🛡️ Crash Prevention**: Global error handling, method validation

## Architecture

### 🏗️ Two-Function Design:
1. **FastAPI Service** (Lightweight, no GPU)
   - Handle HTTP requests & authentication
   - Input validation & error handling
   - Always warm, cost-efficient

2. **Real-ESRGAN Processor** (GPU-enabled)
   - AI image processing with Real-ESRGAN
   - GPU T4 + 8GB memory
   - Only active during processing

## Quick Start

### 1. Prerequisites

Install Modal CLI:
```bash
pip install modal
```

Login to Modal:
```bash
modal token new
```

### 2. Setup Authentication

Generate secure tokens:
```bash
python generate_tokens.py
```

Create or update Modal secret:
```bash
# If secret doesn't exist
modal secret create upscaler-auth VALID_TOKENS="token1,token2,token3"

# If secret already exists
modal secret create upscaler-auth --force VALID_TOKENS="token1,token2,token3"
```

### 3. Deploy

```bash
modal deploy main.py
```

### 4. Setup Models (Optional - Recommended)

For optimal performance, pre-download models to persistent storage:
```bash
python scripts/setup_models.py
```

**Note**: This step is optional. If skipped, models will be automatically downloaded on first use, but with longer initial response time.

After setup, you'll get URL like:
```
https://your-username--image-upscaler-auth-fastapi-app.modal.run
```

### 5. Test

**Python Client:**
```bash
# Update endpoint URL and token in scripts/test_client.py first
python scripts/test_client.py
```

**Web Interface:**
```bash
# Open web_example.html in browser
# Enter your Modal URL and Bearer token
```

**cURL Examples:**

See [cURL Examples](#curl-examples) section below for comprehensive examples.

## API Documentation

### Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| `GET` | `/` | ❌ | API info and endpoints |
| `GET` | `/health` | ❌ | Health check |
| `GET` | `/docs` | ❌ | FastAPI automatic documentation |
| `POST` | `/upscale` | ✅ | Main upscaling endpoint |
| `GET` | `/upscale` | ❌ | Usage info (helpful error) |

### Authentication
All protected endpoints require Bearer token:
```
Authorization: Bearer your-secret-token-here
```

### Upscale Image

**Request:**
```http
POST /upscale
Content-Type: application/json
Authorization: Bearer <token>

{
  "image": "base64_encoded_image_string",
  "scale": 4,
  "output_ext": "png"
}
```

**Response:**
```json
{
  "download_url": "https://your-url/download/file_id",
  "file_id": "unique_file_identifier",
  "original_size": [width, height],
  "upscaled_size": [width, height],
  "expires_at": "2024-01-01T13:00:00Z",
  "output_format": "png"
}
```

**Parameters:**
- `image`: Base64 encoded image (PNG, JPG, JPEG)
- `scale`: Upscaling factor (2 or 4)
- `output_ext`: Output format - `"png"` or `"jpg"` (optional, default: `"png"`)

**Error Responses:**
- `400`: Invalid input (wrong scale, missing image, invalid output format)
- `401`: Invalid or missing Bearer token
- `405`: Wrong HTTP method
- `500`: Processing error

## Output Formats

### PNG (Default)
```json
{
  "output_ext": "png"
}
```
- **Pros**: Lossless quality, supports transparency
- **Cons**: Larger file size
- **Best for**: High-quality images, images with transparency

### JPG/JPEG
```json
{
  "output_ext": "jpg"
}
```
- **Pros**: Smaller file size (60-80% smaller than PNG)
- **Cons**: Lossy compression (95% quality), no transparency
- **Best for**: Photos, web usage, storage optimization
- **Note**: RGBA images automatically converted to RGB

### Format Selection Examples
```bash
# High quality (larger file)
curl -X POST "https://your-url/upscale" \
  -H "Authorization: Bearer token" \
  -d '{"image":"base64","scale":4,"output_ext":"png"}'

# Optimized size (smaller file)  
curl -X POST "https://your-url/upscale" \
  -H "Authorization: Bearer token" \
  -d '{"image":"base64","scale":4,"output_ext":"jpg"}'

# Default (PNG if not specified)
curl -X POST "https://your-url/upscale" \
  -H "Authorization: Bearer token" \
  -d '{"image":"base64","scale":4}'
```

## Files Structure

```
realesrgan-fastapi/
├── main.py                    # 🎯 Entry point
├── core/                      # 🔧 Core components
│   ├── config.py             # ⚙️ Modal setup & configuration
│   ├── upscaler.py           # 🤖 Real-ESRGAN processing
│   └── api.py                # 🌐 FastAPI app (endpoints, models, auth)
├── utils/                     # 🛠️ Utilities
│   └── cleanup.py            # 🗑️ Auto cleanup functions
├── scripts/                   # 📜 Setup & testing scripts
│   ├── setup_models.py       # 📥 Model setup (run once)
│   ├── generate_tokens.py    # 🔐 Token generator
│   └── test_client.py        # 🧪 Python test client
├── web_example.html          # 🌐 Web interface
├── requirements.txt          # 📦 Dependencies
└── README.md                # 📖 Documentation
```

## Security Features

✅ **Bearer Token Authentication**  
✅ **Multiple tokens support** (comma-separated)  
✅ **Secure token storage** (Modal secrets)  
✅ **Input validation** (scale, image format)  
✅ **Error handling** (prevents crash loops)  
✅ **Method validation** (GET/POST handling)  

## Technical Details

### Model Information
- **Model**: Real-ESRGAN x4plus (4x), x2plus (2x)
- **Input Formats**: PNG, JPG, JPEG (base64 encoded)
- **Output Formats**: PNG (lossless) or JPG (smaller size, 95% quality)
- **Max Scale**: 4x upscaling
- **Model Size**: ~65MB (auto-downloaded)

### Performance
- **Cold Start (with pre-cached models)**: 3-5 seconds
- **Cold Start (without setup)**: 10-15 seconds (auto-download models)
- **Warm Requests**: 2-3 seconds per image
- **GPU**: NVIDIA T4 (8GB VRAM)
- **Memory**: 8GB allocated
- **Timeout**: 5 minutes per request
- **Model Storage**: Persistent Modal Volume with auto-fallback

### Cost Optimization
- **FastAPI service**: Always warm, minimal cost
- **GPU processing**: Only active during image processing
- **Persistent storage**: Models cached in Modal Volume (no re-download)
- **Auto-scaling**: Based on request volume
- **Fast cold starts**: 3-5 seconds vs 10-15 seconds
- **Serverless pricing**: Pay per actual usage

## Troubleshooting

### Common Issues

**1. Secret already exists error:**
```bash
modal secret create upscaler-auth --force VALID_TOKENS="your-tokens"
```

**2. OpenCV/GPU errors:**
- Fixed with proper system dependencies (libgl1-mesa-glx)
- GPU image includes all required libraries

**3. Authentication errors:**
- Verify token in Modal secrets dashboard
- Check Authorization header format: `Bearer <token>`

**4. Slow first request:**
- Run `python setup_models.py` to pre-cache models
- Without setup, first request auto-downloads models (slower)

**5. CUDA out of memory errors:**
- Current optimizations: tiling (512px), FP16 precision, auto-resize (max 2048px)
- For larger images, upgrade GPU in `main.py` line ~85:
  - `gpu="T4"` (14GB) - Good for 2K images
  - `gpu="A10G"` (24GB) - Better for 4K images  
  - `gpu="A100"` (40GB) - Best for 8K+ images
- Then redeploy: `modal deploy main.py`

**6. Large image timeouts:**
- Current timeout: 5 minutes
- For very large images, consider resizing input first

### Monitoring

Check deployment status:
```bash
modal app list
modal app logs image-upscaler-auth
```

View secrets:
```bash
modal secret list
```

## cURL Examples

### Health Check (No Authentication)
```bash
curl -X GET "https://your-username--image-upscaler-auth-fastapi-app.modal.run/health"
```

### API Information
```bash
curl -X GET "https://your-username--image-upscaler-auth-fastapi-app.modal.run/"
```

### Upscale Image
```bash
# First, encode your image to base64
base64 -i input.jpg > image_base64.txt

# Send upscale request (PNG output - default)
curl -X POST "https://your-username--image-upscaler-auth-fastapi-app.modal.run/upscale" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "image": "'$(cat image_base64.txt)'",
    "scale": 4,
    "output_ext": "png"
  }'

# Or request JPG output (smaller file size)
curl -X POST "https://your-username--image-upscaler-auth-fastapi-app.modal.run/upscale" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "image": "'$(cat image_base64.txt)'",
    "scale": 4,
    "output_ext": "jpg"
  }'
```

### Save Upscaled Result
```bash
# Complete workflow: upscale and download result
RESPONSE=$(curl -s -X POST "https://your-username--image-upscaler-auth-fastapi-app.modal.run/upscale" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "image": "'$(base64 -i input.jpg | tr -d '\n')'",
    "scale": 4,
    "output_ext": "jpg"
  }')

# Extract download URL and download the file
DOWNLOAD_URL=$(echo $RESPONSE | jq -r '.download_url')
curl -o upscaled_result.jpg "$DOWNLOAD_URL"
```

### Test Authentication
```bash
# Test with invalid token (should return 401)
curl -X POST "https://your-username--image-upscaler-auth-fastapi-app.modal.run/upscale" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer invalid-token" \
  -d '{
    "image": "dummy_base64_string",
    "scale": 4
  }'
```


## Development

### Local Testing
```bash
# Generate tokens
python scripts/generate_tokens.py

# Update test client with your URL and token
python scripts/test_client.py

# Or use web interface
open web_example.html
```

### Adding Features
- **Rate limiting**: Add per-token request limits
- **Usage tracking**: Log requests per token
- **Multiple models**: Support different AI models
- **Binary upload**: Alternative to base64 for large files

## License

MIT License - Feel free to use and modify!