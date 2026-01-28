"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import Mock, patch
import tempfile
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from handwriting_api.config.settings import Settings
from handwriting_api.core.ocr_engine import OCREngine
from handwriting_api.utils.image_validator import ImageValidator


@pytest.fixture
def test_settings():
    """Test settings with safe defaults."""
    return Settings(
        debug=True,
        ocr_use_gpu=False,  # Use CPU for tests
        max_image_size=1024 * 1024,  # 1MB for tests
        max_image_dimensions=(1024, 1024),
        enable_metrics=False,  # Disable metrics in tests
        log_level="DEBUG"
    )


@pytest.fixture
def mock_ocr_engine():
    """Mock OCR engine for testing."""
    engine = Mock(spec=OCREngine)
    engine.get_model_info.return_value = {
        "engine": "PaddleOCR",
        "language": "en",
        "use_angle_cls": True,
        "use_gpu": False,
        "model_version": "3.0"
    }
    return engine


@pytest.fixture
def image_validator():
    """Image validator instance."""
    return ImageValidator()


@pytest.fixture
def sample_text_image():
    """Create a sample image with text for testing."""
    # Create a white background image
    image = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to built-in if not available
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw some text
    draw.text((50, 50), "Hello World", fill='black', font=font)
    draw.text((50, 100), "This is a test", fill='black', font=font)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr


@pytest.fixture
def sample_jpeg_image():
    """Create a sample JPEG image for testing."""
    image = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw some shapes and text
    draw.rectangle([100, 100, 700, 500], outline='black', width=2)
    draw.ellipse([200, 200, 600, 400], outline='blue', width=3)
    
    try:
        font = ImageFont.load_default()
        draw.text((250, 250), "Test JPEG Image", fill='red', font=font)
    except:
        pass
    
    # Convert to JPEG bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    img_byte_arr.seek(0)
    
    return img_byte_arr


@pytest.fixture
def invalid_image_bytes():
    """Invalid image bytes for testing validation."""
    return b"This is not an image file"


@pytest.fixture
def oversized_image():
    """Create an oversized image for testing size limits."""
    # Create a large image (this will be larger than our test limit)
    image = Image.new('RGB', (2000, 2000), color='white')
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr


@pytest.fixture
def corrupted_image_bytes():
    """Corrupted image bytes for testing."""
    # Start with a valid PNG header but corrupt the rest
    png_header = b'\x89PNG\r\n\x1a\n'
    corrupted_data = png_header + b'corrupted data that is not valid PNG'
    return corrupted_data


@pytest.fixture
def sample_upload_file(sample_text_image):
    """Create a mock upload file."""
    from fastapi import UploadFile
    
    class MockUploadFile:
        def __init__(self, content, filename="test.png", content_type="image/png"):
            self.content = content
            self.filename = filename
            self.content_type = content_type
            self.file = io.BytesIO(content)
        
        async def read(self):
            return self.content
        
        async def seek(self, pos):
            self.file.seek(pos)
        
        async def close(self):
            pass
    
    return MockUploadFile(sample_text_image.getvalue(), "test.png", "image/png")


@pytest.fixture
def mock_paddleocr():
    """Mock PaddleOCR for testing without actual OCR."""
    with patch('handwriting_api.core.ocr_engine.PaddleOCR') as mock:
        # Configure the mock
        mock_instance = Mock()
        mock_instance.ocr.return_value = [
            [
                # Mock OCR result: coordinates and text+confidence
                [
                    [[100, 100], [200, 100], [200, 150], [100, 150]],  # coordinates
                    ["Hello World", 0.95]  # text and confidence
                ],
                [
                    [[100, 200], [300, 200], [300, 250], [100, 250]],
                    ["This is a test", 0.87]
                ]
            ]
        ]
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability check."""
    with patch('handwriting_api.core.ocr_engine.paddle') as mock_paddle:
        mock_paddle.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 1
        yield True


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU unavailability."""
    with patch('handwriting_api.core.ocr_engine.paddle') as mock_paddle:
        mock_paddle.is_compiled_with_cuda.return_value = False
        mock_paddle.device.cuda.device_count.return_value = 0
        yield False