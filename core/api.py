"""
FastAPI application and endpoints
"""
from .config import app, web_image
from .upscaler import process_upscale, get_file_content

# FastAPI web service (lightweight, no GPU)
@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name("upscaler-auth")]
)
@modal.asgi_app()
def fastapi_app():
    import os
    from fastapi import FastAPI, HTTPException, Request, Depends, status
    from fastapi.responses import JSONResponse, Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    from typing import List
    
    # Authentication (defined inside function to avoid import issues)
    security = HTTPBearer()
    
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify Bearer token against valid tokens from environment"""
        # Get valid tokens from environment (set in Modal secrets)
        VALID_TOKENS = os.getenv("VALID_TOKENS", "your-secret-token-here,another-token").split(",")
        
        token = credentials.credentials
        if token not in VALID_TOKENS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token
    
    # Request/Response models (defined inside function to avoid import issues)
    class UpscaleRequest(BaseModel):
        image: str  # base64 encoded image
        scale: int = 4  # upscale factor (2 or 4)

    class UpscaleResponse(BaseModel):
        download_url: str  # URL to download result
        file_id: str  # Unique file identifier
        original_size: List[int]
        upscaled_size: List[int]
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
    
    @web_app.get("/download/{file_id}")
    async def download_file(file_id: str):
        """Download upscaled image file"""
        try:
            # Call the function that has volume access
            result = get_file_content.remote(file_id)
            
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            
            # Return file as response
            return Response(
                content=result["content"],
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename={result['filename']}"
                }
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
                "GET /download/{file_id}": "Download result file",
                "GET /health": "Health check",
                "GET /docs": "API documentation",
                "GET /": "This info page"
            },
            "authentication": "Bearer token required for /upscale endpoint"
        }
    
    return web_app