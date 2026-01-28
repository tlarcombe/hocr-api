#!/usr/bin/env python3
"""
Demonstration script for the Handwriting Recognition API.
Shows how to use the API with sample images and different endpoints.
"""

import requests
import json
import time
from PIL import Image, ImageDraw, ImageFont
import io
import sys
from pathlib import Path

def create_sample_handwriting_image():
    """Create a sample image with handwriting-like text."""
    # Create a white background image
    image = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw some handwritten-like text
    texts = [
        "Hello World!",
        "This is a handwriting sample",
        "The quick brown fox",
        "jumps over the lazy dog",
        "1234567890"
    ]
    
    y_position = 50
    for i, text in enumerate(texts):
        # Simulate handwriting by varying position slightly
        x_position = 50 + (i * 10)  # Slight stagger
        draw.text((x_position, y_position), text, fill='black', font=font)
        y_position += 60
    
    # Add some lines and shapes to make it more realistic
    draw.line([(40, 30), (560, 30)], fill='gray', width=1)
    draw.line([(40, 350), (560, 350)], fill='gray', width=1)
    
    return image

def test_health():
    """Test the health endpoint."""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   GPU available: {data['gpu_available']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Make sure the server is running: python -m uvicorn handwriting_api.api.main:app --host 127.0.0.1 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\n🔍 Testing Model Information...")
    try:
        response = requests.get("http://127.0.0.1:8000/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Engine: {data['engine']}")
            print(f"   Version: {data['version']}")
            print(f"   Language: {data['language']}")
            print(f"   Supported languages: {', '.join(data['supported_languages'][:5])}...")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_single_image():
    """Test single image recognition."""
    print("\n🔍 Testing Single Image Recognition...")
    
    # Create sample image
    image = create_sample_handwriting_image()
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    try:
        files = {'image': ('handwriting_sample.png', img_byte_arr, 'image/png')}
        data = {
            'return_confidence': 'true',
            'preprocess': 'true'
        }
        
        print("   Sending image for recognition...")
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/recognize",
            files=files,
            data=data,
            timeout=30
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recognition successful ({processing_time:.2f}s)")
            print(f"   Detected text:")
            for line in result['text'].split('\n'):
                print(f"     • {line}")
            print(f"   Confidence scores:")
            for i, item in enumerate(result['results'][:3]):  # Show first 3
                print(f"     • '{item['text']}': {item['confidence']:.2f}")
            if len(result['results']) > 3:
                print(f"     • ... and {len(result['results']) - 3} more")
            return True
        else:
            print(f"❌ Recognition failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Recognition error: {e}")
        return False

def test_batch_processing():
    """Test batch processing with multiple images."""
    print("\n🔍 Testing Batch Processing...")
    
    # Create multiple sample images
    images = []
    for i in range(3):
        image = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((50, 30), f"Sample {i+1}", fill='black', font=font)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        images.append((f"sample_{i+1}.png", img_byte_arr))
    
    try:
        # Prepare files for batch upload
        files = []
        for filename, img_data in images:
            files.append(('images', (filename, img_data, 'image/png')))
        
        data = {
            'return_confidence': 'true',
            'preprocess': 'true'
        }
        
        print("   Sending batch request...")
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/recognize/batch",
            files=files,
            data=data,
            timeout=60
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch processing successful ({processing_time:.2f}s)")
            print(f"   Processed {result['total_images']} images")
            print(f"   Successful: {result['successful']}")
            print(f"   Failed: {result['failed']}")
            
            for i, img_result in enumerate(result['results']):
                status = "✓" if img_result['success'] else "✗"
                print(f"   {status} {img_result['filename']}: {img_result['text']}")
            
            return True
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    print("\n1. Health Check:")
    print("   curl http://127.0.0.1:8000/health")
    
    print("\n2. Model Information:")
    print("   curl http://127.0.0.1:8000/model-info")
    
    print("\n3. Single Image Recognition:")
    print("   curl -X POST http://127.0.0.1:8000/recognize \\")
    print("     -F \"image=@your_image.jpg\" \\")
    print("     -F \"return_confidence=true\" \\")
    print("     -F \"preprocess=true\"")
    
    print("\n4. Batch Processing:")
    print("   curl -X POST http://127.0.0.1:8000/recognize/batch \\")
    print("     -F \"images=@image1.jpg\" \\")
    print("     -F \"images=@image2.png\" \\")
    print("     -F \"return_confidence=true\"")
    
    print("\n5. Python Example:")
    print("""
   import requests
   
   with open('handwritten_note.jpg', 'rb') as f:
       response = requests.post(
           'http://127.0.0.1:8000/recognize',
           files={'image': f},
           data={'return_confidence': 'true'}
       )
   
   result = response.json()
   print(f\"Recognized text: {result['text']}\")
   print(f\"Confidence: {result['results'][0]['confidence']}\")
   """)

def main():
    """Main demonstration function."""
    print("🚀 Handwriting Recognition API Demo")
    print("=" * 50)
    print("This demo will test all API endpoints and show usage examples.")
    print("Make sure the API server is running on http://127.0.0.1:8000")
    print()
    
    # Check if server is running
    if not test_health():
        print("\n❌ Server is not running. Please start it with:")
        print("   python -m uvicorn handwriting_api.api.main:app --host 127.0.0.1 --port 8000")
        return 1
    
    # Run all tests
    tests = [
        test_model_info,
        test_single_image,
        test_batch_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    # Show usage examples
    show_usage_examples()
    
    print(f"\n🎉 Demo completed!")
    print(f"\n📖 API Documentation: http://127.0.0.1:8000/docs (when debug mode is enabled)")
    print(f"🔍 Health Check: http://127.0.0.1:8000/health")
    print(f"📈 Model Info: http://127.0.0.1:8000/model-info")
    
    if passed == total:
        print("\n✅ All tests passed! The API is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the server logs for details.")
        return 1

if __name__ == '__main__':
    exit(main())