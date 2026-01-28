"""
Handwriting Recognition API Implementation Example
Using FastAPI and PaddleOCR v3.0
Production-ready implementation with comprehensive error handling
"""

import io
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import PaddleOCR - this would need to be installed
# pip install paddleocr
from paddleocr import PaddleOCR

app = FastAPI(
    title="Handwriting Recognition API",
    description="Production-ready API for handwritten text recognition",
    version="1.0.0"
)

# Global OCR instance
ocr_engine = None

def initialize_ocr():
    """Initialize OCR engine with optimal settings for handwriting recognition"""
    global ocr_engine
    try:
        # Initialize PaddleOCR with handwriting-optimized settings
        ocr_engine = PaddleOCR(
            use_angle_cls=True,           # Enable angle classification
            lang='en',                    # English language
            use_gpu=False,                # Set to True if GPU available
            det_model_dir=None,           # Use default detection model
            rec_model_dir=None,           # Use default recognition model
            cls_model_dir=None,           # Use default classification model
            enable_mkldnn=True,           # Enable Intel MKL-DNN for CPU optimization
            use_tensorrt=False,           # Set to True for TensorRT optimization
            use_fp16=False,               # Set to True for FP16 inference
            rec_batch_num=1,              # Batch size for recognition
            det_limit_side_len=960,       # Max side length for detection
            det_limit_type='max',         # Limit type for detection
            rec_image_shape='3, 48, 320'  # Recognition input shape
        )
        logger.info("OCR engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OCR engine: {str(e)}")
        raise

class OCRRequest(BaseModel):
    """Request model for OCR processing"""
    language: str = "en"
    preprocessing: bool = True
    confidence_threshold: float = 0.5
    
class OCRResponse(BaseModel):
    """Response model for OCR processing"""
    success: bool
    text: str
    confidence: float
    processing_time: float
    boxes: List[List[List[int]]]
    words: List[Dict[str, Any]]
    error: Optional[str] = None

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Comprehensive image preprocessing pipeline for handwriting recognition
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image optimized for OCR
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Resize if image is too small or too large
        height, width = processed.shape
        if height < 300 or width < 300:
            # Scale up small images
            scale_factor = max(300/height, 300/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            processed = cv2.resize(processed, (new_width, new_height), 
                                 interpolation=cv2.INTER_CUBIC)
        
        # Convert back to 3-channel for PaddleOCR
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        logger.info(f"Image preprocessing completed. Shape: {processed_rgb.shape}")
        return processed_rgb
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return image  # Return original if preprocessing fails

def process_ocr_result(ocr_result: List) -> Dict[str, Any]:
    """
    Process OCR results into structured format
    
    Args:
        ocr_result: Raw OCR result from PaddleOCR
        
    Returns:
        Structured OCR result
    """
    if not ocr_result or not ocr_result[0]:
        return {
            "text": "",
            "confidence": 0.0,
            "boxes": [],
            "words": []
        }
    
    boxes = []
    words = []
    full_text = []
    total_confidence = 0.0
    valid_words = 0
    
    for line in ocr_result[0]:
        if len(line) == 2:  # Format: [box, (text, confidence)]
            box, (text, confidence) = line
            boxes.append(box)
            words.append({
                "text": text,
                "confidence": float(confidence),
                "box": box
            })
            full_text.append(text)
            
            if confidence > 0:  # Only count valid confidences
                total_confidence += confidence
                valid_words += 1
    
    avg_confidence = total_confidence / valid_words if valid_words > 0 else 0.0
    
    return {
        "text": " ".join(full_text),
        "confidence": avg_confidence,
        "boxes": boxes,
        "words": words
    }

@app.on_event("startup")
async def startup_event():
    """Initialize OCR engine on startup"""
    logger.info("Starting Handwriting Recognition API...")
    initialize_ocr()

@app.get("/")
async def root():
    """API health check endpoint"""
    return {
        "message": "Handwriting Recognition API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "ocr_engine": "initialized" if ocr_engine is not None else "not_initialized",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recognize", response_model=OCRResponse)
async def recognize_handwriting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "en",
    preprocessing: bool = True,
    confidence_threshold: float = 0.5
):
    """
    Recognize handwritten text from uploaded image
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        language: Language for OCR (default: en)
        preprocessing: Enable image preprocessing (default: True)
        confidence_threshold: Minimum confidence threshold (default: 0.5)
        
    Returns:
        OCR result with text, confidence, and bounding boxes
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an image file."
            )
        
        # Read and validate image
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400, 
                detail="File too large. Maximum size is 10MB."
            )
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array for OpenCV processing
        image_array = np.array(image)
        
        # Apply preprocessing if requested
        if preprocessing:
            processed_image = preprocess_image(image_array)
        else:
            # Ensure image is in correct format
            if len(image_array.shape) == 2:  # Grayscale
                processed_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:  # RGBA
                processed_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            else:  # RGB
                processed_image = image_array
        
        # Perform OCR
        ocr_result = ocr_engine.ocr(processed_image, cls=True)
        
        # Process results
        result_data = process_ocr_result(ocr_result)
        
        # Apply confidence threshold
        if result_data["confidence"] < confidence_threshold:
            logger.warning(f"Low confidence result: {result_data['confidence']}")
        
        processing_time = time.time() - start_time
        
        # Log for monitoring
        logger.info(f"OCR completed in {processing_time:.2f}s, "
                   f"confidence: {result_data['confidence']:.2f}, "
                   f"text length: {len(result_data['text'])}")
        
        return OCRResponse(
            success=True,
            text=result_data["text"],
            confidence=result_data["confidence"],
            processing_time=processing_time,
            boxes=result_data["boxes"],
            words=result_data["words"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"OCR processing failed: {str(e)}")
        return OCRResponse(
            success=False,
            text="",
            confidence=0.0,
            processing_time=processing_time,
            boxes=[],
            words=[],
            error=str(e)
        )

@app.post("/recognize/batch")
async def recognize_batch(
    files: List[UploadFile] = File(...),
    language: str = "en",
    preprocessing: bool = True,
    confidence_threshold: float = 0.5
):
    """
    Batch handwriting recognition for multiple images
    
    Args:
        files: List of image files
        language: Language for OCR
        preprocessing: Enable image preprocessing
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        List of OCR results
    """
    results = []
    
    for file in files:
        try:
            # Process each file individually
            result = await recognize_handwriting(
                BackgroundTasks(),
                file,
                language,
                preprocessing,
                confidence_threshold
            )
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "batch_size": len(files),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models():
    """List available OCR models and their status"""
    return {
        "current_model": "PaddleOCR v3.0 (PP-OCRv5)",
        "language_support": ["en", "ch", "japan", "korean", "fr", "german", "es"],
        "status": "active",
        "capabilities": [
            "handwriting_recognition",
            "text_detection",
            "angle_classification",
            "multilingual_support"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)