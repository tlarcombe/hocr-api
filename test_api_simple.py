#!/usr/bin/env python3
"""Simple test for the handwriting recognition API."""

import requests
import json
import time
from PIL import Image
import io

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data['status']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_model_info():
    """Test model info endpoint."""
    print("Testing model info endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8000/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Model info available: {data['engine']} v{data['version']}")
            return True
        else:
            print(f"✗ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Model info error: {e}")
        return False

def test_recognize():
    """Test recognition endpoint."""
    print("Testing recognition endpoint...")
    
    # Create a simple test image
    image = Image.new('RGB', (400, 200), color='white')
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    try:
        files = {'image': ('test.png', img_byte_arr, 'image/png')}
        data = {'return_confidence': 'true', 'preprocess': 'true'}
        
        response = requests.post(
            "http://127.0.0.1:8000/recognize",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Recognition successful")
            print(f"  Text: {result['text']}")
            print(f"  Results: {len(result['results'])} regions detected")
            print(f"  Processing time: {result['processing_time']:.2f}s")
            return True
        else:
            print(f"✗ Recognition failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Recognition error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Handwriting Recognition API")
    print("=" * 40)
    
    # Wait a moment for server to be ready
    print("Waiting for server to start...")
    time.sleep(2)
    
    tests = [
        test_health,
        test_model_info,
        test_recognize
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the server logs for details.")
        return 1

if __name__ == '__main__':
    exit(main())