"""Test image validation functionality."""

import pytest
from PIL import Image
import io
import os

from handwriting_api.utils.image_validator import ImageValidator, ImageValidationError
from handwriting_api.config.settings import settings


class TestImageValidator:
    """Test image validator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create image validator instance."""
        return ImageValidator()
    
    def create_test_image(self, format='PNG', size=(400, 300), mode='RGB'):
        """Create a test image."""
        image = Image.new(mode, size, color='white')
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def test_validate_valid_png(self, validator):
        """Test validation of valid PNG image."""
        test_image = self.create_test_image('PNG')
        
        # Should not raise any exception
        validator.validate_image_file(test_image.getvalue(), "test.png")
    
    def test_validate_valid_jpeg(self, validator):
        """Test validation of valid JPEG image."""
        test_image = self.create_test_image('JPEG')
        
        # Should not raise any exception
        validator.validate_image_file(test_image.getvalue(), "test.jpg")
    
    def test_validate_valid_webp(self, validator):
        """Test validation of valid WebP image."""
        test_image = self.create_test_image('WEBP')
        
        # Should not raise any exception
        validator.validate_image_file(test_image.getvalue(), "test.webp")
    
    def test_validate_invalid_file_extension(self, validator):
        """Test validation with invalid file extension."""
        test_image = self.create_test_image('PNG')
        
        with pytest.raises(ImageValidationError) as exc_info:
            validator.validate_image_file(test_image.getvalue(), "test.bmp")
        
        assert "Invalid file extension" in str(exc_info.value)
    
    def test_validate_invalid_format(self, validator):
        """Test validation with invalid image format."""
        # Create a BMP image (which we don't support)
        test_image = self.create_test_image('BMP')
        
        with pytest.raises(ImageValidationError) as exc_info:
            validator.validate_image_file(test_image.getvalue(), "test.png")
        
        assert "Invalid image format" in str(exc_info.value)
    
    def test_validate_oversized_image(self, validator):
        """Test validation of oversized image."""
        # Create a large image that exceeds size limits
        test_image = self.create_test_image('PNG', size=(2000, 2000))
        
        # Make sure it's larger than our test limit (1MB)
        image_data = test_image.getvalue()
        if len(image_data) < settings.max_image_size:
            pytest.skip("Image not large enough for size validation test")
        
        with pytest.raises(ImageValidationError) as exc_info:
            validator.validate_image_file(image_data, "test.png")
        
        assert "Image file too large" in str(exc_info.value)
    
    def test_validate_too_small_image(self, validator):
        """Test validation of too small image."""
        # Create a very small image
        test_image = self.create_test_image('PNG', size=(5, 5))
        
        with pytest.raises(ImageValidationError) as exc_info:
            validator.validate_image_file(test_image.getvalue(), "test.png")
        
        assert "Image dimensions too small" in str(exc_info.value)
    
    def test_validate_corrupted_image(self, validator):
        """Test validation of corrupted image."""
        # Create corrupted image data
        corrupted_data = b'\x89PNG\r\n\x1a\n' + b'corrupted data'
        
        with pytest.raises(ImageValidationError) as exc_info:
            validator.validate_image_file(corrupted_data, "test.png")
        
        assert "corrupted" in str(exc_info.value).lower()
    
    def test_validate_non_image_data(self, validator):
        """Test validation of non-image data."""
        text_data = b"This is not an image file"
        
        with pytest.raises(ImageValidationError) as exc_info:
            validator.validate_image_file(text_data, "test.png")
        
        assert "Cannot identify image file format" in str(exc_info.value)
    
    def test_load_and_validate_valid_image(self, validator):
        """Test loading and validating valid image."""
        test_image = self.create_test_image('PNG')
        
        image = validator.load_and_validate_image(test_image.getvalue(), "test.png")
        
        assert isinstance(image, Image.Image)
        assert image.format == 'PNG'
        assert image.mode in ['RGB', 'L']
    
    def test_load_and_validate_rgba_image(self, validator):
        """Test loading and validating RGBA image."""
        test_image = self.create_test_image('PNG', mode='RGBA')
        
        image = validator.load_and_validate_image(test_image.getvalue(), "test.png")
        
        assert isinstance(image, Image.Image)
        assert image.mode in ['RGB', 'L']  # Should be converted
    
    def test_load_and_validate_invalid_image(self, validator):
        """Test loading and validating invalid image."""
        invalid_data = b"Not an image"
        
        with pytest.raises(ImageValidationError):
            validator.load_and_validate_image(invalid_data, "test.png")
    
    def test_get_image_info_rgb(self, validator):
        """Test getting image info for RGB image."""
        test_image = self.create_test_image('PNG', size=(800, 600))
        image = Image.open(io.BytesIO(test_image.getvalue()))
        
        info = validator.get_image_info(image)
        
        assert info["format"] == "PNG"
        assert info["mode"] == "RGB"
        assert info["size"]["width"] == 800
        assert info["size"]["height"] == 600
        assert info["size"]["total_pixels"] == 800 * 600
        assert info["has_transparency"] is False
        assert info["is_animated"] is False
        assert info["frames"] == 1
    
    def test_get_image_info_rgba(self, validator):
        """Test getting image info for RGBA image."""
        test_image = self.create_test_image('PNG', mode='RGBA')
        image = Image.open(io.BytesIO(test_image.getvalue()))
        
        info = validator.get_image_info(image)
        
        assert info["has_transparency"] is True
    
    def test_optimize_image_for_ocr(self, validator):
        """Test image optimization for OCR."""
        test_image = self.create_test_image('PNG', mode='RGBA')
        image = Image.open(io.BytesIO(test_image.getvalue()))
        
        optimized = validator.optimize_image_for_ocr(image)
        
        assert isinstance(optimized, Image.Image)
        assert optimized.mode in ['RGB', 'L']
    
    def test_optimize_image_with_orientation(self, validator):
        """Test image optimization with orientation correction."""
        # Create an image and add EXIF data with orientation
        test_image = self.create_test_image('JPEG')
        image = Image.open(io.BytesIO(test_image.getvalue()))
        
        # Mock EXIF data (this is a simplified test)
        image.info['exif'] = b'mock_exif_data'
        
        optimized = validator.optimize_image_for_ocr(image)
        
        assert isinstance(optimized, Image.Image)
    
    def test_allowed_extensions(self, validator):
        """Test allowed extensions list."""
        extensions = validator._get_allowed_extensions()
        
        expected = ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif']
        assert set(extensions) == set(expected)
    
    def test_allowed_formats(self, validator):
        """Test allowed formats list."""
        formats = validator._get_allowed_formats()
        
        expected = ['JPEG', 'PNG', 'WEBP', 'HEIC', 'HEIF']
        assert set(formats) == set(expected)
    
    def test_extension_validation_edge_cases(self, validator):
        """Test edge cases for extension validation."""
        # No extension
        assert validator._is_valid_extension("testfile") is False
        
        # Multiple dots
        assert validator._is_valid_extension("test.file.png") is True
        
        # Uppercase extension
        assert validator._is_valid_extension("test.PNG") is True
        
        # Mixed case
        assert validator._is_valid_extension("test.JpG") is True


class TestImageValidatorEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def validator(self):
        """Create image validator instance."""
        return ImageValidator()
    
    def test_empty_file(self, validator):
        """Test validation of empty file."""
        with pytest.raises(ImageValidationError):
            validator.validate_image_file(b"", "test.png")
    
    def test_very_large_filename(self, validator):
        """Test validation with very long filename."""
        long_filename = "a" * 1000 + ".png"
        test_image = validator.create_test_image('PNG')
        
        # Should not raise an exception
        validator.validate_image_file(test_image.getvalue(), long_filename)
    
    def test_unicode_filename(self, validator):
        """Test validation with Unicode filename."""
        unicode_filename = "测试图片.png"
        test_image = validator.create_test_image('PNG')
        
        # Should not raise an exception
        validator.validate_image_file(test_image.getvalue(), unicode_filename)
    
    def test_truncated_image(self, validator):
        """Test validation of truncated image."""
        # Create a valid PNG but truncate it
        test_image = validator.create_test_image('PNG')
        image_data = test_image.getvalue()
        truncated_data = image_data[:len(image_data)//2]
        
        with pytest.raises(ImageValidationError):
            validator.validate_image_file(truncated_data, "test.png")
    
    def test_zero_size_image(self, validator):
        """Test validation of zero-size image."""
        # This is tricky to create, so we'll test the validation logic instead
        # by mocking the PIL Image.open method
        with patch('PIL.Image.open') as mock_open:
            mock_image = Mock()
            mock_image.size = (0, 0)
            mock_image.format = 'PNG'
            mock_image.mode = 'RGB'
            mock_image.verify.return_value = None
            mock_open.return_value.__enter__.return_value = mock_image
            
            with pytest.raises(ImageValidationError) as exc_info:
                validator.validate_image_file(b"fake_image_data", "test.png")
            
            assert "Image dimensions too small" in str(exc_info.value)


class TestImageValidatorPerformance:
    """Test image validator performance characteristics."""
    
    @pytest.fixture
    def validator(self):
        """Create image validator instance."""
        return ImageValidator()
    
    def create_large_test_image(self, size=(1000, 1000)):
        """Create a large test image."""
        image = Image.new('RGB', size, color='white')
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def test_validation_performance(self, validator):
        """Test validation performance with reasonably sized images."""
        import time
        
        test_image = self.create_large_test_image()
        image_data = test_image.getvalue()
        
        start_time = time.time()
        validator.validate_image_file(image_data, "test.png")
        end_time = time.time()
        
        # Validation should be reasonably fast (less than 1 second for a 1000x1000 image)
        assert end_time - start_time < 1.0
    
    def test_multiple_validations(self, validator):
        """Test multiple consecutive validations."""
        test_image = self.create_test_image('PNG')
        image_data = test_image.getvalue()
        
        # Validate the same image multiple times
        for i in range(10):
            validator.validate_image_file(image_data, f"test{i}.png")
        
        # All validations should succeed
        assert True  # If we get here, all validations passed
    
    def create_test_image(self, format='PNG', size=(400, 300), mode='RGB'):
        """Create a test image."""
        image = Image.new(mode, size, color='white')
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        
        return img_byte_arr