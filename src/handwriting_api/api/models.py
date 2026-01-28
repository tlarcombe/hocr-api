"""API request/response models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class OCRResult(BaseModel):
    """OCR result model."""
    text: str = Field(..., description="Recognized text")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    coordinates: Optional[List[List[float]]] = Field(None, description="Bounding box coordinates")


class OCRResponse(BaseModel):
    """OCR response model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "text": "Hello World\nThis is handwritten text",
            "results": [
                {
                    "text": "Hello World",
                    "confidence": 0.95,
                    "coordinates": [[100, 100], [200, 100], [200, 150], [100, 150]]
                },
                {
                    "text": "This is handwritten text",
                    "confidence": 0.87,
                    "coordinates": [[100, 200], [300, 200], [300, 250], [100, 250]]
                }
            ],
            "processing_time": 1.23,
            "model_info": {
                "engine": "PaddleOCR",
                "language": "en",
                "use_gpu": True,
                "model_version": "3.0"
            }
        }
    })
    
    success: bool = Field(..., description="Operation success status")
    text: str = Field(..., description="Extracted text content")
    results: List[OCRResult] = Field(..., description="Detailed OCR results")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: Dict[str, Any] = Field(..., description="OCR model information")


class HealthResponse(BaseModel):
    """Health check response model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "timestamp": "2024-01-28T10:30:00Z",
            "version": "1.0.0",
            "model_loaded": True,
            "gpu_available": True,
            "uptime_seconds": 3600
        }
    })
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether OCR model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": "Invalid image format",
            "message": "Only JPEG, PNG, WebP, and HEIC formats are supported",
            "details": {
                "provided_format": "BMP",
                "supported_formats": ["JPEG", "PNG", "WebP", "HEIC"]
            }
        }
    })
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ModelInfoResponse(BaseModel):
    """Model information response model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "engine": "PaddleOCR",
            "version": "3.0",
            "language": "en",
            "use_angle_cls": True,
            "use_gpu": True,
            "supported_languages": ["en", "ch", "fr", "de", "es", "it", "ja", "ko", "ru"],
            "model_configs": {
                "detection": "DB (Differentiable Binarization)",
                "recognition": "SVTR (Scene Text Recognition)",
                "classification": "PP-LCNet"
            }
        }
    })
    
    engine: str = Field(..., description="OCR engine name")
    version: str = Field(..., description="OCR engine version")
    language: str = Field(..., description="Current language setting")
    use_angle_cls: bool = Field(..., description="Whether angle classification is enabled")
    use_gpu: bool = Field(..., description="Whether GPU acceleration is enabled")
    supported_languages: List[str] = Field(..., description="List of supported languages")
    model_configs: Dict[str, str] = Field(..., description="Model configuration details")


class BatchOCRRequest(BaseModel):
    """Batch OCR request model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "return_confidence": True,
            "preprocess": True
        }
    })
    
    return_confidence: bool = Field(default=True, description="Whether to return confidence scores")
    preprocess: bool = Field(default=True, description="Whether to apply image preprocessing")


class BatchOCRResponse(BaseModel):
    """Batch OCR response model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "results": [
                {
                    "filename": "image1.jpg",
                    "text": "Hello World",
                    "confidence": 0.95,
                    "processing_time": 1.23
                }
            ],
            "total_processing_time": 2.5,
            "total_images": 1,
            "successful": 1,
            "failed": 0
        }
    })
    
    success: bool = Field(..., description="Operation success status")
    results: List[Dict[str, Any]] = Field(..., description="Batch processing results")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    total_images: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successfully processed images")
    failed: int = Field(..., description="Number of failed images")