import requests
import base64
import json
from PIL import Image
import io

def test_upscaler_auth(image_path: str, endpoint_url: str, bearer_token: str):
    """Test the authenticated image upscaler endpoint"""
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    image_base64 = base64.b64encode(image_data).decode()
    
    # Prepare request with Bearer token
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "image": image_base64,
        "scale": 4
    }
    
    print("Sending request to authenticated endpoint...")
    
    # Send request
    response = requests.post(f"{endpoint_url}/upscale", json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        
        # Decode result image
        result_image_data = base64.b64decode(result["upscaled_image"])
        result_image = Image.open(io.BytesIO(result_image_data))
        
        # Save result
        result_image.save("upscaled_result_auth.png")
        
        print(f"✅ Success!")
        print(f"Original size: {result['original_size']}")
        print(f"Upscaled size: {result['upscaled_size']}")
        print(f"Result saved as: upscaled_result_auth.png")
        
    elif response.status_code == 401:
        print("❌ Authentication failed - Invalid token")
        print(response.json())
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def test_without_auth(endpoint_url: str):
    """Test endpoint without authentication (should fail)"""
    
    payload = {
        "image": "dummy_base64",
        "scale": 4
    }
    
    print("\nTesting without authentication (should fail)...")
    
    response = requests.post(f"{endpoint_url}/upscale", json=payload)
    
    if response.status_code == 401:
        print("✅ Correctly rejected - Authentication required")
    else:
        print(f"❌ Unexpected response: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Configuration
    endpoint_url = "https://your-username--image-upscaler-auth-fastapi-app.modal.run"
    bearer_token = "your-secret-token-here"  # Replace with your actual token
    image_path = "test_image.jpg"  # Replace with your test image
    
    print("=== Testing Authenticated Image Upscaler ===")
    
    # Test with authentication
    test_upscaler_auth(image_path, endpoint_url, bearer_token)
    
    # Test without authentication
    test_without_auth(endpoint_url)