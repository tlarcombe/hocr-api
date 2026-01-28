"""OCR engine using Tesseract for handwriting recognition."""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import cv2
import structlog

from ..config.settings import settings


logger = structlog.get_logger(__name__)


class TesseractOCREngine:
    """OCR engine for handwriting recognition using Tesseract OCR."""
    
    def __init__(
        self,
        lang: str = "eng",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
    ):
        """Initialize Tesseract OCR engine.
        
        Args:
            lang: Language code (default: 'eng' for English)
            use_angle_cls: Whether to use angle classification (orientation detection)
            use_gpu: Whether to use GPU acceleration (not supported by Tesseract)
            det_model_dir: Not used in Tesseract (for compatibility)
            rec_model_dir: Not used in Tesseract (for compatibility)
            cls_model_dir: Not used in Tesseract (for compatibility)
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = False  # Tesseract doesn't support GPU
        
        logger.info(
            "Initializing Tesseract OCR engine",
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=self.use_gpu
        )
        
        try:
            # Test if tesseract is available and check available languages
            try:
                pytesseract.get_tesseract_version()
                
                # Check available languages
                available_langs = pytesseract.get_languages()
                logger.info("Available Tesseract languages", languages=available_langs)
                
                if lang not in available_langs:
                    logger.warning(f"Language '{lang}' not available, using first available language", 
                                 available_languages=available_langs)
                    if available_langs:
                        self.lang = available_langs[0]  # Use first available language
                    else:
                        self.lang = None  # Use default
                
                logger.info("Tesseract OCR engine initialized successfully", language=self.lang)
                
            except Exception as e:
                logger.error("Tesseract initialization failed", error=str(e))
                raise RuntimeError(f"Tesseract initialization failed: {e}")
                
        except Exception as e:
            logger.error("Failed to initialize Tesseract OCR engine", error=str(e))
            raise RuntimeError(f"Failed to initialize Tesseract OCR engine: {e}")
    
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
            image = enhancer.enhance(1.3)
            
            # Enhance sharpness
            sharpener = ImageEnhance.Sharpness(image)
            image = sharpener.enhance(1.2)
            
            # Convert to grayscale for better OCR results
            image = ImageOps.grayscale(image)
            
            # Resize if image is too small (Tesseract works better with larger images)
            width, height = image.size
            if width < 600 or height < 400:
                scale_factor = max(600 / width, 400 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Apply noise reduction
            image_array = np.array(image)
            denoised = cv2.fastNlMeansDenoising(image_array)
            image = Image.fromarray(denoised)
            
            # Apply threshold to improve contrast
            _, thresholded = cv2.threshold(np.array(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = Image.fromarray(thresholded)
            
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
        """Recognize handwriting in the given image using Tesseract.
        
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
            
            # Perform OCR using Tesseract
            logger.debug("Starting OCR processing")
            
            # Configure Tesseract for better handwriting recognition
            # Use simpler config that works with available languages
            custom_config = r'--oem 3 --psm 6'
            
            # Get text and confidence data
            if return_confidence:
                # Get detailed OCR data with confidence scores
                lang_param = self.lang if self.lang else None
                data = pytesseract.image_to_data(processed_image, lang=lang_param, config=custom_config, output_type=pytesseract.Output.DICT)
                
                # Process the data to extract text blocks with confidence
                results = []
                n_boxes = len(data['text'])
                
                for i in range(n_boxes):
                    if int(data['conf'][i]) > 0:  # Only include confident detections
                        text = data['text'][i].strip()
                        if text:  # Only include non-empty text
                            confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 scale
                            
                            # Calculate bounding box coordinates
                            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            coordinates = [
                                [x, y],
                                [x + w, y],
                                [x + w, y + h],
                                [x, y + h]
                            ]
                            
                            result = {
                                "text": text,
                                "confidence": confidence,
                                "coordinates": coordinates,
                            }
                            results.append(result)
            else:
                # Simple text extraction without confidence
                lang_param = self.lang if self.lang else None
                text = pytesseract.image_to_string(processed_image, lang=lang_param, config=custom_config)
                text = text.strip()
                
                if text:
                    # Create a single result for the entire image
                    width, height = processed_image.size
                    coordinates = [
                        [0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]
                    ]
                    
                    results = [{
                        "text": text,
                        "confidence": None,
                        "coordinates": coordinates,
                    }]
                else:
                    results = []
            
            processing_time = time.time() - start_time
            logger.info(
                "OCR processing completed",
                results_count=len(results),
                processing_time=processing_time,
                avg_confidence=sum(r["confidence"] for r in results if r["confidence"] is not None) / len(results) if results else 0
            )
            
            return results
            
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
        
        # Combine all detected text
        texts = [result["text"] for result in results if result["text"].strip()]
        return "\n".join(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OCR model.
        
        Returns:
            Model information
        """
        return {
            "engine": "TesseractOCR",
            "language": self.lang,
            "use_angle_cls": self.use_angle_cls,
            "use_gpu": self.use_gpu,
            "model_version": str(pytesseract.get_tesseract_version()),
            "description": "Tesseract OCR engine for handwriting recognition"
        }