"""Image validation and processing utilities."""

import io
from typing import Tuple, Optional
from PIL import Image, ImageFile
import structlog

from ..config.settings import settings


logger = structlog.get_logger(__name__)

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageValidationError(Exception):
    """Custom exception for image validation errors."""
    pass


class ImageValidator:
    """Image validation and preprocessing utilities."""
    
    def __init__(self):
        """Initialize image validator."""
        self.max_size = settings.max_image_size
        self.allowed_types = settings.allowed_image_types
        self.max_dimensions = settings.max_image_dimensions
    
    def validate_image_file(self, file_content: bytes, filename: str) -> None:
        """Validate image file content.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Raises:
            ImageValidationError: If validation fails
        """
        # Check file size
        if len(file_content) > self.max_size:
            raise ImageValidationError(
                f"Image file too large: {len(file_content)} bytes "
                f"(maximum allowed: {self.max_size} bytes)"
            )
        
        # Check file extension
        if not self._is_valid_extension(filename):
            raise ImageValidationError(
                f"Invalid file extension: {filename}. "
                f"Allowed extensions: {', '.join(self._get_allowed_extensions())}"
            )
        
        try:
            # Try to open image with PIL
            with Image.open(io.BytesIO(file_content)) as img:
                # Validate image format
                if not self._is_valid_format(img.format):
                    raise ImageValidationError(
                        f"Invalid image format: {img.format}. "
                        f"Allowed formats: {', '.join(self._get_allowed_formats())}"
                    )
                
                # Validate image dimensions
                width, height = img.size
                max_width, max_height = self.max_dimensions
                
                if width > max_width or height > max_height:
                    raise ImageValidationError(
                        f"Image dimensions too large: {width}x{height} pixels. "
                        f"Maximum allowed: {max_width}x{max_height} pixels"
                    )
                
                if width < 10 or height < 10:
                    raise ImageValidationError(
                        f"Image dimensions too small: {width}x{height} pixels. "
                        f"Minimum required: 10x10 pixels"
                    )
                
                # Check for corrupted images
                try:
                    img.verify()
                except Exception as e:
                    raise ImageValidationError(f"Image file appears to be corrupted: {e}")
                
                logger.debug(
                    "Image validation passed",
                    filename=filename,
                    format=img.format,
                    size=(width, height),
                    file_size=len(file_content)
                )
                
        except Image.UnidentifiedImageError:
            raise ImageValidationError("Cannot identify image file format")
        except Exception as e:
            raise ImageValidationError(f"Image validation failed: {e}")
    
    def load_and_validate_image(self, file_content: bytes, filename: str) -> Image.Image:
        """Load and validate image from file content.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Returns:
            PIL Image object
            
        Raises:
            ImageValidationError: If validation fails
        """
        # Validate first
        self.validate_image_file(file_content, filename)
        
        try:
            # Load image
            image = Image.open(io.BytesIO(file_content))
            
            # Ensure image is in RGB mode for consistent processing
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            logger.info(
                "Image loaded and validated successfully",
                filename=filename,
                format=image.format,
                mode=image.mode,
                size=image.size
            )
            
            return image
            
        except Exception as e:
            raise ImageValidationError(f"Failed to load image: {e}")
    
    def _is_valid_extension(self, filename: str) -> bool:
        """Check if file extension is valid."""
        valid_extensions = {
            '.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'
        }
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        return f'.{file_ext}' in valid_extensions
    
    def _is_valid_format(self, image_format: Optional[str]) -> bool:
        """Check if image format is valid."""
        if not image_format:
            return False
        
        format_mapping = {
            'JPEG': 'image/jpeg',
            'PNG': 'image/png',
            'WEBP': 'image/webp',
            'HEIC': 'image/heic',
            'HEIF': 'image/heif',
        }
        
        mime_type = format_mapping.get(image_format.upper())
        return mime_type in self.allowed_types if mime_type else False
    
    def _get_allowed_extensions(self) -> list:
        """Get list of allowed file extensions."""
        return ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif']
    
    def _get_allowed_formats(self) -> list:
        """Get list of allowed image formats."""
        return ['JPEG', 'PNG', 'WEBP', 'HEIC', 'HEIF']
    
    def get_image_info(self, image: Image.Image) -> dict:
        """Get comprehensive image information.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with image information
        """
        width, height = image.size
        
        info = {
            "format": image.format,
            "mode": image.mode,
            "size": {
                "width": width,
                "height": height,
                "total_pixels": width * height
            },
            "has_transparency": image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info),
            "is_animated": getattr(image, 'is_animated', False),
            "frames": getattr(image, 'n_frames', 1) if hasattr(image, 'n_frames') else 1
        }
        
        # Add EXIF data if available
        if hasattr(image, '_getexif') and image._getexif():
            try:
                exif = image._getexif()
                if exif:
                    info["exif"] = {
                        "orientation": exif.get(0x0112, 1),
                        "software": exif.get(0x0131),
                        "datetime": exif.get(0x0132),
                    }
            except Exception:
                pass  # EXIF data not available or corrupted
        
        return info
    
    def optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Optimize image for better OCR results.
        
        Args:
            image: Input PIL image
            
        Returns:
            Optimized PIL image
        """
        try:
            # Convert to RGB if necessary
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Apply automatic orientation correction if EXIF data is available
            image = self._apply_orientation_correction(image)
            
            logger.debug("Image optimization completed", size=image.size, mode=image.mode)
            return image
            
        except Exception as e:
            logger.warning(f"Image optimization failed, returning original: {e}")
            return image
    
    def _apply_orientation_correction(self, image: Image.Image) -> Image.Image:
        """Apply orientation correction based on EXIF data."""
        try:
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                if exif:
                    orientation = exif.get(0x0112, 1)
                    
                    # Apply rotation based on EXIF orientation
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                        
        except Exception as e:
            logger.debug(f"Orientation correction failed: {e}")
        
        return image