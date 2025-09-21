"""
Authentication utilities
"""
import os
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Security
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