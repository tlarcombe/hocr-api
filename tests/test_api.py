"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io
from PIL import Image

from handwriting_api.api.main import app, ocr_engine
from handwriting_api.config.settings import settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_ocr_engine():
    """Mock OCR engine for API tests."""
    with patch('handwriting_api.api.main.ocr_engine') as mock:
        mock.get_model_info.return_value = {
            "engine": "PaddleOCR",
            "language": "en",
            "use_angle_cls": True,
            "use_gpu": False,
            "model_version": "3.0"
        }
        mock.recognize_handwriting.return_value = [
            {
                "text": "Hello World",
                "confidence": 0.95,
                "coordinates": [[100, 100], [200, 100], [200, 150], [100, 150]]
            }
        ]
        mock.extract_text_only.return_value = "Hello World"
        yield mock


class TestHealthEndpoints:
    """Test health check and info endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "uptime_seconds" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Handwriting Recognition API"
        assert data["version"] == settings.app_version
        assert "docs" in data
        assert "health" in data
    
    def test_model_info(self, client, mock_ocr_engine):
        """Test model info endpoint."""
        with patch('handwriting_api.api.main.ocr_engine', mock_ocr_engine):
            response = client.get("/model-info")
            assert response.status_code == 200
            
            data = response.json()
            assert data["engine"] == "PaddleOCR"
            assert data["version"] == "3.0"
            assert "supported_languages" in data
            assert "model_configs" in data
    
    def test_model_info_no_engine(self, client):
        """Test model info when OCR engine is not available."""
        with patch('handwriting_api.api.main.ocr_engine', None):
            response = client.get("/model-info")
            assert response.status_code == 503
            
            data = response.json()
            assert "OCR engine not initialized" in data["message"]


class TestHandwritingRecognition:
    """Test handwriting recognition endpoints."""
    
    def create_test_image(self, format='PNG'):
        """Create a simple test image."""
        image = Image.new('RGB', (200, 100), color='white')
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def test_recognize_handwriting_success(self, client, mock_ocr_engine):
        """Test successful handwriting recognition."""
        test_image = self.create_test_image()
        
        with patch('handwriting_api.api.main.ocr_engine', mock_ocr_engine):
            response = client.post(
                "/recognize",
                files={"image": ("test.png", test_image, "image/png")},
                data={"return_confidence": "true", "preprocess": "true"}
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["text"] == "Hello World"
            assert len(data["results"]) > 0
            assert "processing_time" in data
            assert "model_info" in data
    
    def test_recognize_handwriting_no_confidence(self, client, mock_ocr_engine):
        """Test recognition without confidence scores."""
        test_image = self.create_test_image()
        
        with patch('handwriting_api.api.main.ocr_engine', mock_ocr_engine):
            response = client.post(
                "/recognize",
                files={"image": ("test.png", test_image, "image/png")},
                data={"return_confidence": "false", "preprocess": "true"}
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["text"] == "Hello World"
    
    def test_recognize_handwriting_no_engine(self, client):
        """Test recognition when OCR engine is not available."""
        test_image = self.create_test_image()
        
        with patch('handwriting_api.api.main.ocr_engine', None):
            response = client.post(
                "/recognize",
                files={"image": ("test.png", test_image, "image/png")}
            )
            
            assert response.status_code == 503
            assert "OCR engine not initialized" in response.json()["message"]
    
    def test_recognize_handwriting_invalid_image(self, client):
        """Test recognition with invalid image."""
        invalid_image = io.BytesIO(b"This is not an image")
        
        response = client.post(
            "/recognize",
            files={"image": ("test.png", invalid_image, "image/png")}
        )
        
        assert response.status_code == 400
        assert "Image validation failed" in response.json()["message"]
    
    def test_recognize_handwriting_no_image(self, client):
        """Test recognition without image."""
        response = client.post("/recognize")
        
        assert response.status_code == 422  # Validation error
    
    def test_recognize_handwriting_wrong_format(self, client):
        """Test recognition with wrong file format."""
        text_file = io.BytesIO(b"This is text, not an image")
        
        response = client.post(
            "/recognize",
            files={"image": ("test.txt", text_file, "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_batch_recognize_success(self, client, mock_ocr_engine):
        """Test successful batch recognition."""
        test_image1 = self.create_test_image()
        test_image2 = self.create_test_image()
        
        with patch('handwriting_api.api.main.ocr_engine', mock_ocr_engine):
            response = client.post(
                "/recognize/batch",
                files=[
                    ("images", ("test1.png", test_image1, "image/png")),
                    ("images", ("test2.png", test_image2, "image/png"))
                ],
                data={"return_confidence": "true", "preprocess": "true"}
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["total_images"] == 2
            assert data["successful"] == 2
            assert data["failed"] == 0
            assert len(data["results"]) == 2
            
            for result in data["results"]:
                assert result["success"] is True
                assert "filename" in result
                assert "text" in result
                assert "processing_time" in result
    
    def test_batch_recognize_too_many_images(self, client):
        """Test batch recognition with too many images."""
        test_image = self.create_test_image()
        
        # Create 11 images (exceeds limit)
        files = []
        for i in range(11):
            files.append(("images", (f"test{i}.png", test_image, "image/png")))
        
        response = client.post("/recognize/batch", files=files)
        
        assert response.status_code == 400
        assert "Maximum 10 images allowed" in response.json()["message"]
    
    def test_batch_recognize_mixed_success(self, client, mock_ocr_engine):
        """Test batch recognition with mixed success/failure."""
        test_image = self.create_test_image()
        invalid_image = io.BytesIO(b"Invalid image data")
        
        with patch('handwriting_api.api.main.ocr_engine', mock_ocr_engine):
            response = client.post(
                "/recognize/batch",
                files=[
                    ("images", ("valid.png", test_image, "image/png")),
                    ("images", ("invalid.png", invalid_image, "image/png"))
                ]
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert data["total_images"] == 2
            assert data["successful"] == 1
            assert data["failed"] == 1
            assert data["success"] is False  # Not all succeeded


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_image_validation_error(self, client):
        """Test image validation error handling."""
        # Create a file that's too small to be a valid image
        tiny_file = io.BytesIO(b"tiny")
        
        response = client.post(
            "/recognize",
            files={"image": ("test.png", tiny_file, "image/png")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Image validation failed"
        assert "message" in data
    
    def test_ocr_processing_error(self, client, mock_ocr_engine):
        """Test OCR processing error handling."""
        # Make the OCR engine raise an exception
        mock_ocr_engine.recognize_handwriting.side_effect = Exception("OCR processing failed")
        
        test_image = Image.new('RGB', (200, 100), color='white')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        with patch('handwriting_api.api.main.ocr_engine', mock_ocr_engine):
            response = client.post(
                "/recognize",
                files={"image": ("test.png", img_byte_arr, "image/png")}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "OCR processing failed" in data["message"]
    
    def test_unsupported_image_format(self, client):
        """Test unsupported image format handling."""
        # Create a BMP image (which we don't support)
        test_image = Image.new('RGB', (200, 100), color='white')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='BMP')
        img_byte_arr.seek(0)
        
        response = client.post(
            "/recognize",
            files={"image": ("test.bmp", img_byte_arr, "image/bmp")}
        )
        
        assert response.status_code == 400
        assert "Invalid file extension" in response.json()["message"]


class TestMetrics:
    """Test metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns data."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        
        # Check that some basic metrics are present
        content = response.text
        assert "handwriting_api_requests_total" in content
        assert "handwriting_api_request_duration_seconds" in content


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        
        # CORS should allow the origin
        assert response.headers.get("access-control-allow-origin") == "*"
    
    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/recognize",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        assert "POST" in response.headers.get("access-control-allow-methods", "")