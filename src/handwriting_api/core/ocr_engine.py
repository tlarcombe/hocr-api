"""OCR engine using PaddleOCR for handwriting recognition."""

import io
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from paddleocr import PaddleOCR
import structlog

from ..config.settings import settings


logger = structlog.get_logger(__name__)


class OCREngine:
    """OCR engine for handwriting recognition using PaddleOCR."""
    
    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = True,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
    ):
        """Initialize OCR engine.
        
        Args:
            lang: Language code (default: 'en')
            use_angle_cls: Whether to use angle classification
            use_gpu: Whether to use GPU acceleration
            det_model_dir: Directory for detection model
            rec_model_dir: Directory for recognition model
            cls_model_dir: Directory for classification model
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu and self._check_gpu_availability()
        
        logger.info(
            "Initializing OCR engine",
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=self.use_gpu
        )
        
        try:
            # Initialize PaddleOCR with the new API
            ocr_params = {
                'lang': lang,
                'use_doc_orientation_classify': use_angle_cls,
                'text_detection_model_dir': det_model_dir,
                'text_recognition_model_dir': rec_model_dir,
                'doc_orientation_classify_model_dir': cls_model_dir,
                'text_recognition_batch_size': 1,
                'text_rec_score_thresh': 0.5,
                'return_word_box': False,
            }
            
            # Remove None values
            ocr_params = {k: v for k, v in ocr_params.items() if v is not None}
            
            self.ocr = PaddleOCR(**ocr_params)
            logger.info("OCR engine initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize OCR engine", error=str(e))
            raise RuntimeError(f"Failed to initialize OCR engine: {e}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for OCR processing."""
        try:
            import paddle
            return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except ImportError:
            logger.warning("Paddle GPU support not available, falling back to CPU")
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

            # Resize if image is too small (PaddleOCR works better with larger images)
            width, height = image.size
            if width < 800 or height < 600:
                scale_factor = max(800 / width, 600 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Apply noise reduction (for RGB images)
            image_array = np.array(image)
            denoised = cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)
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
        """Recognize handwriting in the given image.
        
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
            
            # Convert PIL image to numpy array
            image_array = np.array(processed_image)
            
            # Perform OCR
            logger.debug("Starting OCR processing")
            ocr_results = self.ocr.predict(
                image_array,
                use_doc_orientation_classify=self.use_angle_cls,
                text_rec_score_thresh=0.5,
                return_word_box=False
            )

            if not ocr_results:
                logger.info("No text detected in image")
                return []

            ocr_result = ocr_results[0]
            # PaddleOCR 3.x returns OCRResult objects - access data via json property
            json_data = ocr_result.json if hasattr(ocr_result, 'json') else ocr_result
            res = json_data.get('res', json_data) if isinstance(json_data, dict) else {}
            rec_texts = res.get('rec_texts', [])
            rec_scores = res.get('rec_scores', [])
            dt_polys = res.get('dt_polys', [])

            if not rec_texts:
                logger.info("No text detected in image")
                return []

            # Process results
            results = []
            for i, text in enumerate(rec_texts):
                confidence = float(rec_scores[i]) if return_confidence and i < len(rec_scores) else None
                coordinates = dt_polys[i] if i < len(dt_polys) else []

                result = {
                    "text": str(text),
                    "confidence": confidence,
                    "coordinates": coordinates,
                }
                results.append(result)
            
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

        def get_min_y(coords):
            """Get minimum y-coordinate from polygon coordinates."""
            if not coords:
                return 0
            return min(point[1] for point in coords if len(point) >= 2)

        def get_avg_y(coords):
            """Get average y-coordinate from polygon coordinates."""
            if not coords:
                return 0
            y_coords = [point[1] for point in coords if len(point) >= 2]
            return sum(y_coords) / len(y_coords) if y_coords else 0

        def get_min_x(coords):
            """Get minimum x-coordinate from polygon coordinates."""
            if not coords:
                return 0
            return min(point[0] for point in coords if len(point) >= 2)

        # Sort results by vertical position (y-coordinate)
        sorted_results = sorted(results, key=lambda r: get_min_y(r.get("coordinates", [])))

        lines = []
        current_line = []
        current_y = None
        line_threshold = 50  # Pixels threshold for line grouping

        for result in sorted_results:
            coords = result.get("coordinates", [])
            avg_y = get_avg_y(coords)

            if current_y is None or abs(avg_y - current_y) <= line_threshold:
                current_line.append(result)
                if current_y is None:
                    current_y = avg_y
            else:
                # Sort current line by x-coordinate and extract text
                if current_line:
                    current_line.sort(key=lambda r: get_min_x(r.get("coordinates", [])))
                    lines.append([r["text"] for r in current_line])

                current_line = [result]
                current_y = avg_y

        # Add the last line
        if current_line:
            current_line.sort(key=lambda r: get_min_x(r.get("coordinates", [])))
            lines.append([r["text"] for r in current_line])

        return lines
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OCR model.
        
        Returns:
            Model information
        """
        return {
            "engine": "PaddleOCR",
            "language": self.lang,
            "use_angle_cls": self.use_angle_cls,
            "use_gpu": self.use_gpu,
            "model_version": "3.0",
        }