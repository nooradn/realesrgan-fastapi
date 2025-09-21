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

After deployment, you'll get URL like:
```
https://your-username--image-upscaler-auth-fastapi-app.modal.run
```

### 4. Test

**Python Client:**
```bash
# Update endpoint URL and token in test_client.py first
python test_client.py
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
  "scale": 4
}
```

**Response:**
```json
{
  "upscaled_image": "base64_encoded_result_string",
  "original_size": [width, height],
  "upscaled_size": [width, height]
}
```

**Parameters:**
- `image`: Base64 encoded image (PNG, JPG, JPEG)
- `scale`: Upscaling factor (2 or 4)

**Error Responses:**
- `400`: Invalid input (wrong scale, missing image)
- `401`: Invalid or missing Bearer token
- `405`: Wrong HTTP method
- `500`: Processing error

## Files Structure

```
realesrgan-fastapi/
├── main.py              # 🔒 FastAPI app + GPU processor
├── test_client.py       # 🧪 Python test client
├── web_example.html     # 🌐 Web interface for testing
├── generate_tokens.py   # 🔐 Secure token generator
├── requirements.txt     # 📦 Dependencies
└── README.md           # 📖 This documentation
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
- **Input Formats**: PNG, JPG, JPEG
- **Output Format**: PNG (base64 encoded)
- **Max Scale**: 4x upscaling
- **Model Size**: ~65MB (auto-downloaded)

### Performance
- **Cold Start**: 10-15 seconds (first request, model download)
- **Warm Requests**: 2-5 seconds per image
- **GPU**: NVIDIA T4 (8GB VRAM)
- **Memory**: 8GB allocated
- **Timeout**: 5 minutes per request

### Cost Optimization
- **FastAPI service**: Always warm, minimal cost
- **GPU processing**: Only active during image processing
- **Auto-scaling**: Based on request volume
- **Model caching**: Faster subsequent requests
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

**4. Large image timeouts:**
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

# Send upscale request
curl -X POST "https://your-username--image-upscaler-auth-fastapi-app.modal.run/upscale" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "image": "'$(cat image_base64.txt)'",
    "scale": 4
  }'
```

### Save Upscaled Result
```bash
# Complete workflow: upscale and save result
curl -X POST "https://your-username--image-upscaler-auth-fastapi-app.modal.run/upscale" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "image": "'$(base64 -i input.jpg | tr -d '\n')'",
    "scale": 4
  }' | jq -r '.upscaled_image' | base64 -d > upscaled_output.png
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
python generate_tokens.py

# Update test client with your URL and token
python test_client.py

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