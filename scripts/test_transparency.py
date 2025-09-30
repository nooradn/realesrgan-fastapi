#!/usr/bin/env python3
"""
Test script to verify PNG transparency is preserved during upscaling
"""

import base64
import io
import requests
from PIL import Image, ImageDraw
import numpy as np

def create_test_png_with_transparency():
    """Create a test PNG image with transparency"""
    # Create a 100x100 RGBA image
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    
    # Draw a red circle with some transparency
    draw.ellipse([20, 20, 80, 80], fill=(255, 0, 0, 180))  # Semi-transparent red
    
    # Draw a blue rectangle with full opacity
    draw.rectangle([10, 10, 50, 50], fill=(0, 0, 255, 255))  # Opaque blue
    
    return img

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_local_transparency():
    """Test transparency preservation locally (without API call)"""
    print("🧪 Testing PNG transparency preservation locally...")
    
    # Create test image
    test_img = create_test_png_with_transparency()
    print(f"✅ Created test image: {test_img.size}, mode: {test_img.mode}")
    
    # Save original for comparison
    test_img.save("test_original.png")
    print("✅ Saved original test image as test_original.png")
    
    # Convert to base64
    img_base64 = image_to_base64(test_img)
    print(f"✅ Converted to base64: {len(img_base64)} characters")
    
    # Test the transparency detection logic
    img_array = np.array(test_img)
    has_transparency = test_img.mode in ('RGBA', 'LA') or 'transparency' in test_img.info
    
    print(f"✅ Transparency detected: {has_transparency}")
    print(f"✅ Image mode: {test_img.mode}")
    print(f"✅ Image shape: {img_array.shape}")
    
    if has_transparency and test_img.mode == 'RGBA':
        alpha_channel = img_array[:, :, 3]
        unique_alpha = np.unique(alpha_channel)
        print(f"✅ Alpha channel values: {unique_alpha}")
        print(f"✅ Has partial transparency: {len(unique_alpha) > 2 or (len(unique_alpha) == 2 and 0 in unique_alpha and 255 in unique_alpha)}")

def test_api_transparency(api_url, auth_token):
    """Test transparency preservation via API"""
    print(f"🌐 Testing PNG transparency via API: {api_url}")
    
    # Create test image
    test_img = create_test_png_with_transparency()
    img_base64 = image_to_base64(test_img)
    
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "image": img_base64,
        "scale": 2,
        "output_ext": "png"
    }
    
    try:
        # Make API request
        print("📤 Sending upscale request...")
        response = requests.post(f"{api_url}/upscale", json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upscale successful: {result}")
            
            # Download the result
            download_url = result["download_url"]
            print(f"📥 Downloading result from: {download_url}")
            
            download_response = requests.get(download_url)
            if download_response.status_code == 200:
                # Save and analyze result
                with open("test_upscaled.png", "wb") as f:
                    f.write(download_response.content)
                
                # Load and check transparency
                result_img = Image.open("test_upscaled.png")
                print(f"✅ Result image: {result_img.size}, mode: {result_img.mode}")
                
                if result_img.mode == 'RGBA':
                    result_array = np.array(result_img)
                    alpha_channel = result_array[:, :, 3]
                    unique_alpha = np.unique(alpha_channel)
                    print(f"✅ Result alpha values: {unique_alpha}")
                    print("🎉 Transparency preserved!")
                else:
                    print("❌ Transparency lost - image is not RGBA")
            else:
                print(f"❌ Download failed: {download_response.status_code}")
        else:
            print(f"❌ API request failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    # Test locally first
    test_local_transparency()
    
    print("\n" + "="*50)
    print("To test via API, run:")
    print("python scripts/test_transparency.py --api-url YOUR_API_URL --token YOUR_TOKEN")
    print("="*50)
    
    # Uncomment and modify these lines to test via API
    # API_URL = "https://your-modal-app-url.modal.run"
    # AUTH_TOKEN = "your-secret-token"
    # test_api_transparency(API_URL, AUTH_TOKEN)