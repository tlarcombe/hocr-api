"""
Comprehensive test suite for Handwriting Recognition API
Tests all endpoints and core functionality
"""

import pytest
import asyncio
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handwriting_api_example import app, initialize_ocr, preprocess_image

client = TestClient(app)

class TestHandwritingAPI:
    """Test suite for Handwriting Recognition API"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample handwritten image for testing"""
        # Create a simple handwritten-like image
        img_array = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        # Add some text-like patterns (simulating handwriting)
        cv2 = pytest.importorskip("cv2")
        cv2.putText(img_array, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (0, 0, 0), 3)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
    
    @pytest.fixture
    def invalid_file(self):
        """Create an invalid file for testing"""
        return io.BytesIO(b"This is not an image file")
    
    def test_root_endpoint(self):
        """Test the root health endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Handwriting Recognition API is running"
        assert "version" in data
        assert "timestamp" in data
    
    def test_health_endpoint(self):
        """Test the detailed health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_models_endpoint(self):
        """Test the models information endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "current_model" in data
        assert "language_support" in data
        assert "capabilities" in data
    
    @pytest.mark.asyncio
    async def test_recognize_endpoint_valid_image(self, sample_image):
        """Test OCR recognition with valid image"""
        files = {"file": ("test.png", sample_image, "image/png")}
        
        response = client.post("/recognize", files=files)
        
        # Should return 200 even if OCR fails
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "text" in data
        assert "confidence" in data
        assert "processing_time" in data
        assert "boxes" in data
        assert "words" in data
        
        # Basic validation
        assert isinstance(data["success"], bool)
        assert isinstance(data["text"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["processing_time"], float)
        assert isinstance(data["boxes"], list)
        assert isinstance(data["words"], list)
    
    def test_recognize_endpoint_invalid_file(self, invalid_file):
        """Test OCR recognition with invalid file"""
        files = {"file": ("test.txt", invalid_file, "text/plain")}
        
        response = client.post("/recognize", files=files)
        
        # Should return 400 for invalid file type
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid file type" in data["detail"]
    
    def test_recognize_endpoint_no_file(self):
        """Test OCR recognition without file"""
        response = client.post("/recognize")
        
        # Should return 422 for missing file
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_batch_recognize_endpoint(self, sample_image):
        """Test batch OCR recognition"""
        files = [
            ("files", ("test1.png", sample_image, "image/png")),
            ("files", ("test2.png", sample_image, "image/png"))
        ]
        
        response = client.post("/recognize/batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "batch_size" in data
        assert "results" in data
        assert "timestamp" in data
        
        assert data["batch_size"] == 2
        assert len(data["results"]) == 2
        
        for result in data["results"]:
            assert "filename" in result
            assert "result" in result or "error" in result
    
    def test_preprocessing_function(self):
        """Test image preprocessing function"""
        try:
            cv2 = pytest.importorskip("cv2")
            
            # Create test image
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 128
            
            # Test preprocessing
            processed = preprocess_image(test_image)
            
            # Check output format
            assert isinstance(processed, np.ndarray)
            assert len(processed.shape) == 3  # Should be RGB
            assert processed.shape[2] == 3    # 3 channels
            
        except ImportError:
            pytest.skip("OpenCV not available")
    
    def test_large_file_rejection(self):
        """Test that large files are rejected"""
        # Create a large fake file (11MB)
        large_file = io.BytesIO(b"0" * (11 * 1024 * 1024))
        files = {"file": ("large.png", large_file, "image/png")}
        
        response = client.post("/recognize", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "File too large" in data["detail"]
    
    def test_different_image_formats(self):
        """Test with different image formats"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        
        formats_to_test = ['PNG', 'JPEG', 'BMP']
        
        for fmt in formats_to_test:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=fmt)
            img_bytes.seek(0)
            
            files = {"file": (f"test.{fmt.lower()}", img_bytes, f"image/{fmt.lower()}")}
            response = client.post("/recognize", files=files)
            
            # Should accept valid image formats
            assert response.status_code == 200
    
    def test_grayscale_image(self):
        """Test with grayscale image"""
        # Create grayscale image
        img = Image.new('L', (100, 100), color=128)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"file": ("test_gray.png", img_bytes, "image/png")}
        response = client.post("/recognize", files=files)
        
        assert response.status_code == 200
    
    def test_rgba_image(self):
        """Test with RGBA image"""
        # Create RGBA image
        img = Image.new('RGBA', (100, 100), color=(255, 255, 255, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"file": ("test_rgba.png", img_bytes, "image/png")}
        response = client.post("/recognize", files=files)
        
        assert response.status_code == 200
    
    @pytest.mark.parametrize("confidence_threshold", [0.1, 0.5, 0.8, 0.9])
    def test_confidence_threshold_parameter(self, sample_image, confidence_threshold):
        """Test different confidence threshold values"""
        files = {"file": ("test.png", sample_image, "image/png")}
        params = {"confidence_threshold": confidence_threshold}
        
        response = client.post("/recognize", files=files, params=params)
        
        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
    
    @pytest.mark.parametrize("preprocessing", [True, False])
    def test_preprocessing_parameter(self, sample_image, preprocessing):
        """Test preprocessing parameter"""
        files = {"file": ("test.png", sample_image, "image/png")}
        params = {"preprocessing": preprocessing}
        
        response = client.post("/recognize", files=files, params=params)
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        def make_request():
            files = {"file": ("test.png", img_bytes.copy(), "image/png")}
            return client.post("/recognize", files=files)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for result in results:
            assert result.status_code == 200
    
    def test_error_handling(self):
        """Test various error scenarios"""
        
        # Test with corrupted image data
        corrupted_data = io.BytesIO(b"\x89PNG\x00\x00\x00\x00")
        files = {"file": ("corrupted.png", corrupted_data, "image/png")}
        response = client.post("/recognize", files=files)
        
        # Should handle gracefully
        assert response.status_code == 200
        data = response.json()
        assert "error" in data or data["success"] == False

class TestPerformance:
    """Performance tests for the API"""
    
    def test_response_time(self, sample_image):
        """Test API response time"""
        import time
        
        files = {"file": ("test.png", sample_image, "image/png")}
        
        start_time = time.time()
        response = client.post("/recognize", files=files)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        # Response should be reasonably fast (adjust threshold as needed)
        assert response_time < 30  # 30 seconds max
        
        # Check processing time in response
        data = response.json()
        assert "processing_time" in data
        assert data["processing_time"] < 30

if __name__ == "__main__":
    pytest.main([__file__, "-v"])