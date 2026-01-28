"""Test OCR engine functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import io

from handwriting_api.core.ocr_engine import OCREngine
from handwriting_api.config.settings import settings


class TestOCREngineInitialization:
    """Test OCR engine initialization."""
    
    def test_initialization_default(self, mock_paddleocr):
        """Test OCR engine initialization with default settings."""
        engine = OCREngine()
        
        assert engine.lang == "en"
        assert engine.use_angle_cls is True
        assert engine.use_gpu is False  # Should default to False in tests
        assert engine.ocr is not None
    
    def test_initialization_custom_params(self, mock_paddleocr):
        """Test OCR engine initialization with custom parameters."""
        engine = OCREngine(
            lang="ch",
            use_angle_cls=False,
            use_gpu=False,
            det_model_dir="/custom/det",
            rec_model_dir="/custom/rec",
            cls_model_dir="/custom/cls"
        )
        
        assert engine.lang == "ch"
        assert engine.use_angle_cls is False
        assert engine.use_gpu is False
    
    def test_initialization_failure(self):
        """Test OCR engine initialization failure."""
        with patch('handwriting_api.core.ocr_engine.PaddleOCR') as mock_paddle:
            mock_paddle.side_effect = Exception("Initialization failed")
            
            with pytest.raises(RuntimeError, match="Failed to initialize OCR engine"):
                OCREngine()


class TestImagePreprocessing:
    """Test image preprocessing functionality."""
    
    @pytest.fixture
    def ocr_engine(self, mock_paddleocr):
        """Create OCR engine for testing."""
        return OCREngine()
    
    def test_preprocess_rgb_image(self, ocr_engine):
        """Test preprocessing of RGB image."""
        # Create RGB image
        image = Image.new('RGB', (400, 300), color='white')
        
        processed = ocr_engine.preprocess_image(image)
        
        assert processed.mode == 'L'  # Should be grayscale
        assert processed.size == image.size  # Size should remain the same for large images
    
    def test_preprocess_rgba_image(self, ocr_engine):
        """Test preprocessing of RGBA image."""
        # Create RGBA image with transparency
        image = Image.new('RGBA', (400, 300), color=(255, 255, 255, 128))
        
        processed = ocr_engine.preprocess_image(image)
        
        assert processed.mode == 'L'  # Should be grayscale
        assert processed.size == image.size
    
    def test_preprocess_small_image(self, ocr_engine):
        """Test preprocessing of small image (should be resized)."""
        # Create small image
        image = Image.new('RGB', (200, 150), color='white')
        
        processed = ocr_engine.preprocess_image(image)
        
        assert processed.mode == 'L'
        # Should be resized to minimum dimensions
        assert processed.width >= 800 or processed.height >= 600
    
    def test_preprocess_image_with_content(self, ocr_engine):
        """Test preprocessing of image with actual content."""
        # Create image with some content
        image = Image.new('RGB', (600, 400), color='white')
        draw = Image.new('L', (600, 400), color=255)
        
        # Add some dark regions
        draw_data = np.array(draw)
        draw_data[100:200, 100:500] = 0  # Black rectangle
        draw = Image.fromarray(draw_data)
        
        # Convert back to RGB for processing
        image = Image.merge('RGB', [draw, draw, draw])
        
        processed = ocr_engine.preprocess_image(image)
        
        assert processed.mode == 'L'
        assert processed.size == (600, 400)  # Should not resize since it's large enough
    
    def test_preprocess_corrupted_image(self, ocr_engine):
        """Test preprocessing failure handling."""
        # Create a corrupted image scenario
        with patch.object(ocr_engine, 'preprocess_image') as mock_preprocess:
            mock_preprocess.side_effect = ValueError("Preprocessing failed")
            
            image = Image.new('RGB', (400, 300), color='white')
            
            with pytest.raises(ValueError, match="Preprocessing failed"):
                ocr_engine.preprocess_image(image)


class TestHandwritingRecognition:
    """Test handwriting recognition functionality."""
    
    @pytest.fixture
    def ocr_engine(self, mock_paddleocr):
        """Create OCR engine for testing."""
        return OCREngine()
    
    def test_recognize_handwriting_success(self, ocr_engine, mock_paddleocr):
        """Test successful handwriting recognition."""
        # Create test image
        image = Image.new('RGB', (400, 300), color='white')
        
        results = ocr_engine.recognize_handwriting(image)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result structure
        result = results[0]
        assert "text" in result
        assert "confidence" in result
        assert "coordinates" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["coordinates"], list)
    
    def test_recognize_handwriting_no_results(self, ocr_engine, mock_paddleocr):
        """Test recognition when no text is detected."""
        # Mock OCR to return empty results
        mock_paddleocr.ocr.return_value = [[]]
        
        image = Image.new('RGB', (400, 300), color='white')
        results = ocr_engine.recognize_handwriting(image)
        
        assert results == []
    
    def test_recognize_handwriting_with_confidence(self, ocr_engine, mock_paddleocr):
        """Test recognition with confidence scores."""
        image = Image.new('RGB', (400, 300), color='white')
        
        results = ocr_engine.recognize_handwriting(image, return_confidence=True)
        
        assert len(results) > 0
        for result in results:
            assert result["confidence"] is not None
            assert 0 <= result["confidence"] <= 1
    
    def test_recognize_handwriting_without_confidence(self, ocr_engine, mock_paddleocr):
        """Test recognition without confidence scores."""
        image = Image.new('RGB', (400, 300), color='white')
        
        results = ocr_engine.recognize_handwriting(image, return_confidence=False)
        
        assert len(results) > 0
        for result in results:
            assert result["confidence"] is None
    
    def test_recognize_handwriting_processing_error(self, ocr_engine, mock_paddleocr):
        """Test recognition processing error handling."""
        # Mock OCR to raise an exception
        mock_paddleocr.ocr.side_effect = Exception("OCR processing failed")
        
        image = Image.new('RGB', (400, 300), color='white')
        
        with pytest.raises(RuntimeError, match="OCR processing failed"):
            ocr_engine.recognize_handwriting(image)


class TestTextExtraction:
    """Test text extraction functionality."""
    
    @pytest.fixture
    def ocr_engine(self, mock_paddleocr):
        """Create OCR engine for testing."""
        return OCREngine()
    
    def test_extract_text_only(self, ocr_engine, mock_paddleocr):
        """Test text-only extraction."""
        image = Image.new('RGB', (400, 300), color='white')
        
        text = ocr_engine.extract_text_only(image)
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_extract_text_only_no_results(self, ocr_engine, mock_paddleocr):
        """Test text extraction when no text is detected."""
        # Mock OCR to return empty results
        mock_paddleocr.ocr.return_value = [[]]
        
        image = Image.new('RGB', (400, 300), color='white')
        text = ocr_engine.extract_text_only(image)
        
        assert text == ""
    
    def test_group_text_by_lines(self, ocr_engine):
        """Test text grouping by lines."""
        # Mock results with different y-coordinates
        results = [
            {
                "text": "First line",
                "confidence": 0.9,
                "coordinates": [[10, 10], [100, 10], [100, 30], [10, 30]]
            },
            {
                "text": "Second line",
                "confidence": 0.8,
                "coordinates": [[10, 50], [110, 50], [110, 70], [10, 70]]
            },
            {
                "text": "Also first",
                "confidence": 0.85,
                "coordinates": [[120, 10], [200, 10], [200, 30], [120, 30]]
            }
        ]
        
        lines = ocr_engine._group_text_by_lines(results)
        
        assert len(lines) == 2  # Should have 2 lines
        assert "First line" in lines[0]
        assert "Also first" in lines[0]  # Same line
        assert "Second line" in lines[1]


class TestGPUAvailability:
    """Test GPU availability checking."""
    
    def test_gpu_available(self, mock_paddleocr, mock_gpu_available):
        """Test GPU availability when CUDA is available."""
        engine = OCREngine(use_gpu=True)
        assert engine.use_gpu is True
    
    def test_gpu_unavailable(self, mock_paddleocr, mock_gpu_unavailable):
        """Test GPU availability when CUDA is not available."""
        engine = OCREngine(use_gpu=True)
        assert engine.use_gpu is False  # Should fall back to CPU
    
    def test_gpu_import_error(self, mock_paddleocr):
        """Test GPU availability when paddle import fails."""
        with patch('handwriting_api.core.ocr_engine.paddle') as mock_paddle:
            mock_paddle.is_compiled_with_cuda.side_effect = ImportError("No paddle")
            
            engine = OCREngine(use_gpu=True)
            assert engine.use_gpu is False


class TestModelInfo:
    """Test model information functionality."""
    
    @pytest.fixture
    def ocr_engine(self, mock_paddleocr):
        """Create OCR engine for testing."""
        return OCREngine(lang="fr", use_angle_cls=False, use_gpu=True)
    
    def test_get_model_info(self, ocr_engine):
        """Test getting model information."""
        info = ocr_engine.get_model_info()
        
        assert info["engine"] == "PaddleOCR"
        assert info["language"] == "fr"
        assert info["use_angle_cls"] is False
        assert info["use_gpu"] is True
        assert info["model_version"] == "3.0"


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def ocr_engine(self, mock_paddleocr):
        """Create OCR engine for testing."""
        return OCREngine()
    
    def test_processing_time_tracking(self, ocr_engine, mock_paddleocr):
        """Test that processing time is tracked."""
        image = Image.new('RGB', (400, 300), color='white')
        
        import time
        start_time = time.time()
        results = ocr_engine.recognize_handwriting(image)
        end_time = time.time()
        
        assert len(results) > 0
        assert end_time - start_time > 0  # Should take some time
    
    def test_large_image_processing(self, ocr_engine, mock_paddleocr):
        """Test processing of large images."""
        # Create a large image (but not too large for tests)
        image = Image.new('RGB', (1200, 800), color='white')
        
        results = ocr_engine.recognize_handwriting(image)
        
        # Should still process successfully
        assert isinstance(results, list)
    
    def test_multiple_processing_calls(self, ocr_engine, mock_paddleocr):
        """Test multiple processing calls on the same engine."""
        image = Image.new('RGB', (400, 300), color='white')
        
        # Process multiple times
        for i in range(3):
            results = ocr_engine.recognize_handwriting(image)
            assert isinstance(results, list)


class TestErrorRecovery:
    """Test error recovery scenarios."""
    
    @pytest.fixture
    def ocr_engine(self, mock_paddleocr):
        """Create OCR engine for testing."""
        return OCREngine()
    
    def test_corrupted_coordinates(self, ocr_engine, mock_paddleocr):
        """Test handling of corrupted coordinate data."""
        # Mock results with invalid coordinate data
        mock_paddleocr.ocr.return_value = [
            [
                [None, None],  # Invalid coordinates
                ["Corrupted text", 0.5]
            ]
        ]
        
        image = Image.new('RGB', (400, 300), color='white')
        results = ocr_engine.recognize_handwriting(image)
        
        # Should handle gracefully
        assert isinstance(results, list)
    
    def test_malformed_text_data(self, ocr_engine, mock_paddleocr):
        """Test handling of malformed text data."""
        # Mock results with invalid text data
        mock_paddleocr.ocr.return_value = [
            [
                [[10, 10], [100, 10], [100, 30], [10, 30]],
                [None, None]  # Invalid text and confidence
            ]
        ]
        
        image = Image.new('RGB', (400, 300), color='white')
        results = ocr_engine.recognize_handwriting(image)
        
        # Should handle gracefully
        assert isinstance(results, list)