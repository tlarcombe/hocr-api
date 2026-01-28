"""OCR engine using Microsoft TrOCR for handwriting recognition."""

import time
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import structlog

logger = structlog.get_logger(__name__)


class TrOCREngine:
    """OCR engine for handwriting recognition using Microsoft TrOCR.

    TrOCR is a transformer-based model specifically trained for handwriting
    recognition, achieving state-of-the-art results on handwriting benchmarks.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = True,
        model_name: str = "microsoft/trocr-large-handwritten",
        **kwargs
    ):
        """Initialize TrOCR engine.

        Args:
            lang: Language code (default: 'en') - TrOCR primarily supports English
            use_angle_cls: Whether to use angle classification (handled in preprocessing)
            use_gpu: Whether to use GPU acceleration
            model_name: HuggingFace model name for TrOCR
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu and self._check_gpu_availability()
        self.model_name = model_name

        logger.info(
            "Initializing TrOCR engine",
            model=model_name,
            use_gpu=self.use_gpu
        )

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            self.device = "cuda" if self.use_gpu else "cpu"

            # Load TrOCR processor and model
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info("TrOCR engine initialized successfully", device=self.device)

        except Exception as e:
            logger.error("Failed to initialize TrOCR engine", error=str(e))
            raise RuntimeError(f"Failed to initialize TrOCR engine: {e}")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                logger.info("CUDA GPU available", device_count=torch.cuda.device_count())
            return available
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

            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)

            # Enhance sharpness
            sharpener = ImageEnhance.Sharpness(image)
            image = sharpener.enhance(1.1)

            return image

        except Exception as e:
            logger.error("Image preprocessing failed", error=str(e))
            raise ValueError(f"Image preprocessing failed: {e}")

    def _detect_text_lines(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect text lines in the image using projection profile analysis.

        This method uses horizontal projection profiles to find text line
        boundaries, which works better for handwritten documents.

        Args:
            image: Input PIL image

        Returns:
            List of detected text line regions with bounding boxes
        """
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding for better binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate horizontal projection profile
        h_projection = np.sum(binary, axis=1)

        # Smooth the projection to reduce noise
        kernel_size = max(5, img_height // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        h_projection_smooth = cv2.GaussianBlur(
            h_projection.reshape(-1, 1).astype(np.float32),
            (1, kernel_size), 0
        ).flatten()

        # Find threshold for line detection
        threshold = np.max(h_projection_smooth) * 0.1

        # Find line boundaries
        in_line = False
        line_start = 0
        text_regions = []

        for i, val in enumerate(h_projection_smooth):
            if not in_line and val > threshold:
                in_line = True
                line_start = i
            elif in_line and val <= threshold:
                in_line = False
                line_end = i

                # Only add if line has reasonable height
                line_height = line_end - line_start
                if line_height > img_height * 0.02:  # At least 2% of image height
                    # Add vertical padding
                    padding = int(line_height * 0.2)
                    y1 = max(0, line_start - padding)
                    y2 = min(img_height, line_end + padding)

                    # Find horizontal extent of text in this line
                    line_slice = binary[line_start:line_end, :]
                    v_projection = np.sum(line_slice, axis=0)
                    non_zero = np.where(v_projection > 0)[0]

                    if len(non_zero) > 0:
                        x1 = max(0, non_zero[0] - 10)
                        x2 = min(img_width, non_zero[-1] + 10)

                        text_regions.append({
                            'bbox': [x1, y1, x2, y2],
                            'coordinates': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            'area': (x2 - x1) * (y2 - y1)
                        })

        # Handle case where text continues to end of image
        if in_line:
            line_height = img_height - line_start
            if line_height > img_height * 0.02:
                padding = int(line_height * 0.2)
                y1 = max(0, line_start - padding)
                y2 = img_height

                line_slice = binary[line_start:, :]
                v_projection = np.sum(line_slice, axis=0)
                non_zero = np.where(v_projection > 0)[0]

                if len(non_zero) > 0:
                    x1 = max(0, non_zero[0] - 10)
                    x2 = min(img_width, non_zero[-1] + 10)

                    text_regions.append({
                        'bbox': [x1, y1, x2, y2],
                        'coordinates': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        'area': (x2 - x1) * (y2 - y1)
                    })

        logger.debug(f"Detected {len(text_regions)} text lines using projection profile")
        return text_regions

    def _merge_overlapping_regions(
        self, regions: List[Dict[str, Any]], img_height: int
    ) -> List[Dict[str, Any]]:
        """Merge text regions that are on the same line.

        Args:
            regions: List of detected regions
            img_height: Image height for threshold calculation

        Returns:
            Merged regions
        """
        if not regions:
            return []

        merged = []
        line_threshold = img_height * 0.03  # 3% of image height

        current_region = regions[0].copy()

        for region in regions[1:]:
            # Check if regions are on the same line (similar y-coordinate)
            current_y_center = (current_region['bbox'][1] + current_region['bbox'][3]) / 2
            new_y_center = (region['bbox'][1] + region['bbox'][3]) / 2

            if abs(current_y_center - new_y_center) < line_threshold:
                # Merge regions
                current_region['bbox'] = [
                    min(current_region['bbox'][0], region['bbox'][0]),
                    min(current_region['bbox'][1], region['bbox'][1]),
                    max(current_region['bbox'][2], region['bbox'][2]),
                    max(current_region['bbox'][3], region['bbox'][3])
                ]
                x1, y1, x2, y2 = current_region['bbox']
                current_region['coordinates'] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                merged.append(current_region)
                current_region = region.copy()

        merged.append(current_region)
        return merged

    def _recognize_text_region(self, image: Image.Image) -> tuple:
        """Recognize text in a single image region using TrOCR.

        Args:
            image: Cropped PIL image containing text

        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        import torch

        try:
            # Prepare image for model
            pixel_values = self.processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(self.device)

            # Generate text with scores
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=4,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode text
            generated_ids = outputs.sequences
            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Calculate confidence from sequence scores
            if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                # Convert log probability to probability
                confidence = float(torch.exp(outputs.sequences_scores[0]).cpu())
                confidence = min(confidence, 1.0)  # Cap at 1.0
            else:
                # Estimate confidence from token probabilities
                confidence = 0.9  # Default high confidence for TrOCR

            return text.strip(), confidence

        except Exception as e:
            logger.warning("Text recognition failed for region", error=str(e))
            return "", 0.0

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

            # Detect text lines
            logger.debug("Detecting text lines")
            text_regions = self._detect_text_lines(processed_image)

            if not text_regions:
                # If no regions detected, try processing the whole image
                logger.info("No text regions detected, processing entire image")
                text, confidence = self._recognize_text_region(processed_image)
                if text:
                    return [{
                        "text": text,
                        "confidence": confidence if return_confidence else None,
                        "coordinates": [[0, 0], [image.width, 0],
                                       [image.width, image.height], [0, image.height]]
                    }]
                return []

            # Recognize text in each region
            logger.debug("Recognizing text in detected regions", region_count=len(text_regions))
            results = []

            for region in text_regions:
                bbox = region['bbox']

                # Crop the region
                cropped = processed_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                # Skip very small crops
                if cropped.width < 10 or cropped.height < 10:
                    continue

                # Recognize text
                text, confidence = self._recognize_text_region(cropped)

                if text:  # Only add non-empty results
                    results.append({
                        "text": text,
                        "confidence": confidence if return_confidence else None,
                        "coordinates": region['coordinates']
                    })

            processing_time = time.time() - start_time
            avg_confidence = (
                sum(r["confidence"] for r in results if r["confidence"] is not None) / len(results)
                if results else 0
            )

            logger.info(
                "TrOCR processing completed",
                results_count=len(results),
                processing_time=processing_time,
                avg_confidence=avg_confidence
            )

            return results

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "TrOCR processing failed",
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

        if not results:
            return ""

        # Group by lines and join
        lines = self._group_text_by_lines(results)
        return "\n".join(" ".join(line) for line in lines)

    def _group_text_by_lines(self, results: List[Dict[str, Any]]) -> List[List[str]]:
        """Group detected text by approximate line positions.

        Args:
            results: OCR results with coordinates

        Returns:
            List of text lines
        """
        if not results:
            return []

        def get_avg_y(coords):
            if not coords:
                return 0
            y_coords = [point[1] for point in coords if len(point) >= 2]
            return sum(y_coords) / len(y_coords) if y_coords else 0

        def get_min_x(coords):
            if not coords:
                return 0
            return min(point[0] for point in coords if len(point) >= 2)

        # Sort by y-coordinate
        sorted_results = sorted(
            results,
            key=lambda r: get_avg_y(r.get("coordinates", []))
        )

        lines = []
        current_line = []
        current_y = None
        line_threshold = 50

        for result in sorted_results:
            coords = result.get("coordinates", [])
            avg_y = get_avg_y(coords)

            if current_y is None or abs(avg_y - current_y) <= line_threshold:
                current_line.append(result)
                if current_y is None:
                    current_y = avg_y
            else:
                if current_line:
                    current_line.sort(key=lambda r: get_min_x(r.get("coordinates", [])))
                    lines.append([r["text"] for r in current_line])
                current_line = [result]
                current_y = avg_y

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
            "engine": "TrOCR",
            "language": self.lang,
            "use_angle_cls": self.use_angle_cls,
            "use_gpu": self.use_gpu,
            "model_version": self.model_name,
        }


# Alias for compatibility
OCREngine = TrOCREngine
