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

**Request (Base64 Method):**
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

**Request (URL Method):**
```http
POST /upscale
Content-Type: application/json
Authorization: Bearer <token>

{
  "image_url": "https://example.com/image.jpg",
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
- `image`: Base64 encoded image (PNG, JPG, JPEG) - **either this or image_url**
- `image_url`: Public image URL (PNG, JPG, JPEG) - **either this or image**
- `scale`: Upscaling factor (2 or 4)
- `output_ext`: Output format - `"png"` or `"jpg"` (optional, default: `"png"`)

**Error Responses:**
- `400`: Invalid input (wrong scale, missing image, invalid output format)
- `401`: Invalid or missing Bearer token
- `405`: Wrong HTTP method
- `500`: Processing error

## Input Methods

### Base64 Method
```json
{
  "image": "base64_encoded_string",
  "scale": 4
}
```
- **Pros**: Works with any image, no external dependencies
- **Cons**: Large JSON payload, need to encode first
- **Best for**: Private images, secure uploads, batch processing

### Image URL Method  
```json
{
  "image_url": "https://example.com/image.jpg",
  "scale": 4
}
```
- **Pros**: Simple, small JSON payload, no encoding needed
- **Cons**: Image must be publicly accessible
- **Best for**: Public images, web scraping, quick testing
- **Timeout**: 30 seconds for download
- **User Agent**: Includes proper headers to avoid blocking

### Input Method Examples
```bash
# Base64 method (private/local images)
curl -X POST "https://your-url/upscale" \
  -H "Authorization: Bearer token" \
  -d '{"image":"'$(base64 -i local.jpg)'","scale":4}'

# URL method (public images)
curl -X POST "https://your-url/upscale" \
  -H "Authorization: Bearer token" \
  -d '{"image_url":"https://picsum.photos/800/600","scale":4}'
```

## Transparency Support

### PNG Transparency Preservation
The upscaler now **fully preserves PNG transparency** (alpha channel):

- **Input**: PNG images with transparency (RGBA mode)
- **Processing**: Alpha channel is upscaled separately using high-quality interpolation
- **Output**: PNG with preserved transparency at higher resolution
- **Quality**: No transparency loss during upscaling process

### Transparency Handling by Format
- **PNG → PNG**: ✅ Full transparency preservation (RGBA → RGBA)
- **PNG → JPG**: ⚠️ Transparency converted to white background (RGBA → RGB)
- **JPG → PNG**: ✅ Opaque image in PNG format (RGB → RGB)
- **JPG → JPG**: ✅ Standard JPEG processing (RGB → RGB)

### Testing Transparency
Use the included test script to verify transparency handling:
```bash
python scripts/test_transparency.py
```

## Output Formats

### PNG (Default)
```json
{
  "output_ext": "png"
}
```
- **Pros**: Lossless quality, **full transparency support (RGBA)**
- **Cons**: Larger file size
- **Best for**: High-quality images, images with transparency, logos, graphics
- **Transparency**: Preserves alpha channel from input PNG images

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
├── scripts/                   # �  Setup & testing scripts
│   ├── setup_models.py       # �️ Model setup (run once)
│   ├── generate_tokens.py    # � Toeken generator
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
- **Input Formats**: PNG, JPG, JPEG (base64 encoded or URL)
- **Output Formats**: PNG (lossless, with transparency) or JPG (smaller size, 95% quality)
- **Transparency Support**: Full RGBA preservation for PNG inputs/outputs
- **Max Scale**: 4x upscaling
- **Model Size**: ~65MB (auto-downloaded)

### Performance & Scaling
- **Cold Start (with pre-cached models)**: 3-5 seconds
- **Cold Start (without setup)**: 10-15 seconds (auto-download models)
- **Warm Requests**: 2-3 seconds per image
- **Scale-to-Zero**: Automatic when idle (0 cost during inactivity)
- **Auto-Scaling**: Handles traffic spikes automatically
- **GPU**: NVIDIA T4 (14GB VRAM available, 2GB RAM allocated)
- **Timeout**: 10 minutes per request
- **Model Storage**: Persistent Modal Volume with auto-fallback
- **Concurrent Requests**: Up to 10 GPU instances (configurable)

### Cost Optimization & Auto-Scaling
- **Scale-to-Zero**: Automatically scales down to zero when idle (no cost when unused)
- **Auto-Scaling**: Scales up/down based on request volume automatically
- **FastAPI service**: Lightweight, scales independently from GPU functions
- **GPU processing**: Only active during image processing, scales to zero when idle
- **Persistent storage**: Models cached in Modal Volume (no re-download)
- **Cold Start**: 3-5 seconds for warm containers, 10-15 seconds for cold starts
- **Serverless pricing**: Pay only for actual compute time used
- **Idle Management**: Containers automatically go idle when not processing requests

**Note**: Modal.com handles all scaling automatically. For detailed scaling behavior and configuration options, refer to the [official Modal documentation](https://modal.com/docs).

## Scaling & Infrastructure

### Auto-Scaling Behavior
Modal.com provides automatic scaling with the following features:

- **Scale-to-Zero**: When no requests are being processed, containers automatically scale down to zero, resulting in **zero cost during idle periods**
- **Auto-Scale Up**: Containers automatically spin up when requests arrive
- **Load Balancing**: Multiple containers can run concurrently to handle traffic spikes
- **Idle Detection**: Containers go idle after a period of inactivity and are automatically paused

### Cold Start Considerations
- **First Request**: May take 10-15 seconds if models need to be downloaded
- **Subsequent Requests**: 3-5 seconds if models are cached in persistent storage
- **Warm Containers**: 2-3 seconds for active containers

### Scaling Configuration
Current configuration allows:
- **Max Containers**: 10 concurrent GPU instances
- **Memory per Container**: 2GB RAM (optimized for actual usage)
- **GPU per Container**: NVIDIA T4 (14GB VRAM)

### Cost Implications
- **Pay-per-Use**: Only charged for actual compute time
- **No Idle Costs**: Zero cost when scaled to zero
- **Efficient Resource Usage**: Optimized memory allocation reduces per-request cost

For detailed information about Modal's scaling behavior, container lifecycle, and advanced configuration options, refer to the [official Modal documentation](https://modal.com/docs/guide/lifecycle).

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
- Run `python scripts/setup_models.py` to pre-cache models
- Without setup, first request auto-downloads models (slower)

**5. Storage cleanup:**
- Files expire after 1 hour but are not automatically deleted
- Manual cleanup required via Modal dashboard or CLI
- Monitor storage usage to avoid unexpected costs

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

**Method 1: Base64 Input**
```bash
# First, encode your image to base64
base64 -i input.jpg > image_base64.txt

# Send upscale request
curl -X POST "https://your-username--image-upscaler-auth-fastapi-app.modal.run/upscale" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "image": "'$(cat image_base64.txt)'",
    "scale": 4,
    "output_ext": "png"
  }'
```

**Method 2: Image URL Input**
```bash
# Direct image URL (no encoding needed)
curl -X POST "https://your-username--image-upscaler-auth-fastapi-app.modal.run/upscale" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "image_url": "https://example.com/image.jpg",
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