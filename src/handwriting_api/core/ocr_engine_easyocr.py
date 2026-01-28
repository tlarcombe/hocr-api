"""OCR engine using EasyOCR for handwriting recognition."""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import cv2
import structlog

# Import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from ..config.settings import settings


logger = structlog.get_logger(__name__)


class EasyOCREngine:
    """OCR engine for handwriting recognition using EasyOCR."""
    
    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
    ):
        """Initialize EasyOCR engine.
        
        Args:
            lang: Language code (default: 'en')
            use_angle_cls: Whether to use angle classification (rotation correction)
            use_gpu: Whether to use GPU acceleration
            det_model_dir: Not used in EasyOCR (for compatibility)
            rec_model_dir: Not used in EasyOCR (for compatibility)
            cls_model_dir: Not used in EasyOCR (for compatibility)
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR is not available. Please install it with: pip install easyocr")
        
        logger.info(
            "Initializing EasyOCR engine",
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu
        )
        
        try:
            # Initialize EasyOCR reader
            # EasyOCR supports multiple languages, convert our lang code
            lang_map = {
                'en': ['en'],           # English
                'ch': ['ch_sim', 'ch_tra'],  # Chinese (simplified & traditional)
                'fr': ['fr'],           # French
                'de': ['de'],           # German
                'es': ['es'],           # Spanish
                'it': ['it'],           # Italian
                'ja': ['ja'],           # Japanese
                'ko': ['ko'],           # Korean
                'ru': ['ru'],           # Russian
            }
            
            # Map language code to EasyOCR format
            if lang in lang_map:
                easyocr_langs = lang_map[lang]
            else:
                easyocr_langs = ['en']  # Default to English
            
            logger.info("Creating EasyOCR reader", languages=easyocr_langs, gpu=use_gpu)
            
            self.reader = easyocr.Reader(
                lang_list=easyocr_langs,
                gpu=use_gpu,
                download_enabled=True,  # Download models if needed
                detector=True,          # Enable text detection
                recognizer=True,        # Enable text recognition
                verbose=False           # Reduce output
            )
            
            logger.info("EasyOCR engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize EasyOCR engine", error=str(e))
            raise RuntimeError(f"Failed to initialize EasyOCR engine: {e}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for OCR processing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
            return False
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results.
        
        Args:
            image: Input PIL image
            
        Returns:
            Preprocessed PIL image
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            sharpener = ImageEnhance.Sharpness(image)
            image = sharpener.enhance(1.1)
            
            # Convert to grayscale for better OCR results
            image = ImageOps.grayscale(image)
            
            # Resize if image is too small (EasyOCR works better with larger images)
            width, height = image.size
            if width < 800 or height < 600:
                scale_factor = max(800 / width, 600 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Apply noise reduction
            image_array = np.array(image)
            denoised = cv2.fastNlMeansDenoising(image_array)
            image = Image.fromarray(denoised)
            
            logger.debug("Image preprocessing completed", original_size=(width, height), new_size=image.size)
            return image
            
        except Exception as e:
            logger.error("Image preprocessing failed", error=str(e))
            raise ValueError(f"Image preprocessing failed: {e}")
    
    def recognize_handwriting(
        self, 
        image: Image.Image, 
        return_confidence: bool = True
    ) -> List[Dict[str, Any]]:
        """Recognize handwriting in the given image using EasyOCR.
        
        Args:
            image: Input PIL image
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of recognition results with text and confidence
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Convert PIL image to numpy array (EasyOCR expects numpy arrays)
            image_array = np.array(processed_image)
            
            # Perform OCR using EasyOCR
            logger.debug("Starting OCR processing")
            
            # Read text from image
            results = self.reader.readtext(image_array)
            
            if not results:
                logger.info("No text detected in image")
                return []
            
            # Process EasyOCR results
            processed_results = []
            for (bbox, text, confidence) in results:
                if text.strip():  # Only include non-empty text
                    # Convert bbox to coordinates format
                    coordinates = [[float(point[0]), float(point[1])] for point in bbox]
                    
                    result = {
                        "text": text.strip(),
                        "confidence": float(confidence) if return_confidence else None,
                        "coordinates": coordinates,
                    }
                    processed_results.append(result)
            
            processing_time = time.time() - start_time
            logger.info(
                "OCR processing completed",
                results_count=len(processed_results),
                processing_time=processing_time,
                avg_confidence=sum(r["confidence"] for r in processed_results if r["confidence"] is not None) / len(processed_results) if processed_results else 0
            )
            
            return processed_results
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "OCR processing failed",
                error=str(e),
                processing_time=processing_time
            )
            raise RuntimeError(f"OCR processing failed: {e}")
    
    def extract_text_only(self, image: Image.Image) -> str:
        """Extract only the text content from the image.
        
        Args:
            image: Input PIL image
            
        Returns:
            Extracted text content
        """
        results = self.recognize_handwriting(image, return_confidence=False)
        
        # Combine all detected text with proper spacing
        texts = [result["text"] for result in results if result["text"].strip()]
        
        # Group text by approximate line positions
        if results:
            lines = self._group_text_by_lines(results)
            return "\n".join(" ".join(line) for line in lines)
        else:
            return "\n".join(texts)
    
    def _group_text_by_lines(self, results: List[Dict[str, Any]]) -> List[List[str]]:
        """Group detected text by approximate line positions.
        
        Args:
            results: OCR results with coordinates
            
        Returns:
            List of text lines
        """
        if not results:
            return []
        
        # Sort results by vertical position (y-coordinate)
        sorted_results = sorted(results, key=lambda r: min(coord[1] for coord in r["coordinates"]))
        
        lines = []
        current_line = []
        current_y = None
        line_threshold = 50  # Pixels threshold for line grouping
        
        for result in sorted_results:
            # Calculate average y-coordinate of the bounding box
            if result["coordinates"] and len(result["coordinates"]) > 0:
                y_coords = [coord[1] for coord in result["coordinates"]]
                avg_y = sum(y_coords) / len(y_coords)
            else:
                avg_y = 0
            
            if current_y is None or abs(avg_y - current_y) <= line_threshold:
                current_line.append(result)
                if current_y is None:
                    current_y = avg_y
            else:
                # Sort current line by x-coordinate and extract text
                if current_line:
                    current_line.sort(key=lambda r: min(coord[0] for coord in r["coordinates"]))
                    lines.append([r["text"] for r in current_line])
                
                current_line = [result]
                current_y = avg_y
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda r: min(coord[0] for coord in r["coordinates"]))
            lines.append([r["text"] for r in current_line])
        
        return lines
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OCR model.
        
        Returns:
            Model information
        """
        return {
            "engine": "EasyOCR",
            "language": self.lang,
            "use_angle_cls": self.use_angle_cls,
            "use_gpu": self.use_gpu,
            "model_version": "1.7+",
            "description": "EasyOCR engine for handwriting recognition"
        }