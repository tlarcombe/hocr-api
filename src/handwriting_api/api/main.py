"""Main FastAPI application."""

import time
import asyncio
from datetime import datetime
from typing import Optional, List
import logging

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import structlog
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

from ..config.settings import settings
# Import the EasyOCR engine
from ..core.ocr_engine_easyocr import EasyOCREngine as OCREngine
from ..utils.image_validator import ImageValidator, ImageValidationError
from .models import (
    OCRResponse, 
    HealthResponse, 
    ErrorResponse, 
    ModelInfoResponse,
    BatchOCRRequest,
    BatchOCRResponse
)


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('handwriting_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('handwriting_api_request_duration_seconds', 'API request duration')
ACTIVE_REQUESTS = Gauge('handwriting_api_active_requests', 'Active API requests')
OCR_PROCESSING_TIME = Histogram('handwriting_api_ocr_processing_seconds', 'OCR processing time')
OCR_RESULTS_COUNT = Histogram('handwriting_api_ocr_results_count', 'Number of OCR results per request')

# Global OCR engine instance
ocr_engine: Optional[OCREngine] = None
image_validator = ImageValidator()

# Application start time
app_start_time = datetime.now()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered handwriting recognition API using PaddleOCR",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        ACTIVE_REQUESTS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            process_time = time.time() - start_time
            REQUEST_DURATION.observe(process_time)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        finally:
            ACTIVE_REQUESTS.dec()
    
    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Initialize OCR engine on startup."""
    global ocr_engine
    
    logger.info("Starting handwriting recognition API", version=settings.app_version)
    
    try:
        # Initialize OCR engine
        ocr_engine = OCREngine(
            lang=settings.ocr_lang,
            use_angle_cls=settings.ocr_use_angle_cls,
            use_gpu=settings.ocr_use_gpu,
            det_model_dir=settings.ocr_det_model_dir,
            rec_model_dir=settings.ocr_rec_model_dir,
            cls_model_dir=settings.ocr_cls_model_dir,
        )
        
        logger.info("OCR engine initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize OCR engine", error=str(e))
        raise RuntimeError(f"Failed to initialize OCR engine: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down handwriting recognition API")
    # Add any cleanup logic here


@app.exception_handler(ImageValidationError)
async def image_validation_exception_handler(request: Request, exc: ImageValidationError):
    """Handle image validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Image validation failed",
            message=str(exc),
            details={"validation_error": type(exc).__name__}
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP error",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            message="An unexpected error occurred",
            details={"error_type": type(exc).__name__} if settings.debug else None
        ).model_dump()
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "Handwriting Recognition API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if ocr_engine else "unhealthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        model_loaded=ocr_engine is not None,
        gpu_available=settings.ocr_use_gpu and ocr_engine and ocr_engine.use_gpu if ocr_engine else False,
        uptime_seconds=uptime
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get OCR model information."""
    if not ocr_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR engine not initialized"
        )
    
    model_info = ocr_engine.get_model_info()
    
    return ModelInfoResponse(
        engine=model_info["engine"],
        version=model_info["model_version"],
        language=model_info["language"],
        use_angle_cls=model_info["use_angle_cls"],
        use_gpu=model_info["use_gpu"],
        supported_languages=["en", "ch", "fr", "de", "es", "it", "ja", "ko", "ru"],
        model_configs={
            "detection": "DB (Differentiable Binarization)",
            "recognition": "SVTR (Scene Text Recognition)",
            "classification": "PP-LCNet"
        }
    )


@app.post("/recognize", response_model=OCRResponse)
async def recognize_handwriting(
    image: UploadFile = File(..., description="Image file containing handwriting (JPEG, PNG, WebP, HEIC)"),
    return_confidence: bool = Form(default=True, description="Return confidence scores for each text region"),
    preprocess: bool = Form(default=True, description="Apply image preprocessing for better OCR results")
):
    """Recognize handwriting in uploaded image.
    
    This endpoint accepts an image file and returns recognized handwritten text.
    Supports JPEG, PNG, WebP, and HEIC formats with automatic preprocessing.
    
    - **image**: Image file containing handwriting
    - **return_confidence**: Include confidence scores in results (default: true)
    - **preprocess**: Apply image enhancement preprocessing (default: true)
    """
    if not ocr_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR engine not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Read file content
        file_content = await image.read()
        
        # Validate image
        try:
            pil_image = image_validator.load_and_validate_image(file_content, image.filename)
        except ImageValidationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        # Optimize image for OCR
        if preprocess:
            pil_image = image_validator.optimize_image_for_ocr(pil_image)
        
        # Perform OCR
        ocr_start = time.time()
        ocr_results = ocr_engine.recognize_handwriting(
            pil_image, 
            return_confidence=return_confidence
        )
        ocr_time = time.time() - ocr_start
        
        OCR_PROCESSING_TIME.observe(ocr_time)
        OCR_RESULTS_COUNT.observe(len(ocr_results))
        
        # Format response
        if return_confidence and ocr_results:
            results = [
                {
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "coordinates": result["coordinates"]
                }
                for result in ocr_results
            ]
            extracted_text = "\n".join(result["text"] for result in ocr_results)
        else:
            results = ocr_results if ocr_results else []
            extracted_text = ocr_engine.extract_text_only(pil_image)
        
        total_time = time.time() - start_time
        
        logger.info(
            "Handwriting recognition completed",
            filename=image.filename,
            results_count=len(results),
            processing_time=total_time,
            ocr_time=ocr_time
        )
        
        return OCRResponse(
            success=True,
            text=extracted_text,
            results=results,
            processing_time=total_time,
            model_info=ocr_engine.get_model_info()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Handwriting recognition failed",
            filename=image.filename,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/recognize/batch", response_model=BatchOCRResponse)
async def batch_recognize_handwriting(
    images: List[UploadFile] = File(..., description="Multiple image files for batch processing"),
    request_config: BatchOCRRequest = None
):
    """Batch handwriting recognition for multiple images.
    
    Process multiple images in a single request for improved efficiency.
    """
    if not ocr_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR engine not initialized"
        )
    
    if len(images) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images allowed per batch request"
        )
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    # Use default config if not provided
    if request_config is None:
        request_config = BatchOCRRequest()
    
    for image in images:
        image_start = time.time()
        
        try:
            # Read and validate image
            file_content = await image.read()
            pil_image = image_validator.load_and_validate_image(file_content, image.filename)
            
            # Optimize if requested
            if request_config.preprocess:
                pil_image = image_validator.optimize_image_for_ocr(pil_image)
            
            # Perform OCR
            if request_config.return_confidence:
                ocr_results = ocr_engine.recognize_handwriting(pil_image, return_confidence=True)
                extracted_text = "\n".join(result["text"] for result in ocr_results)
                confidence = sum(r["confidence"] for r in ocr_results if r["confidence"] is not None) / len(ocr_results) if ocr_results else 0
            else:
                extracted_text = ocr_engine.extract_text_only(pil_image)
                confidence = None
            
            results.append({
                "filename": image.filename,
                "text": extracted_text,
                "confidence": confidence,
                "processing_time": time.time() - image_start,
                "success": True
            })
            successful += 1
            
        except Exception as e:
            results.append({
                "filename": image.filename,
                "text": "",
                "confidence": None,
                "processing_time": time.time() - image_start,
                "success": False,
                "error": str(e)
            })
            failed += 1
            
            logger.warning(
                "Batch processing failed for image",
                filename=image.filename,
                error=str(e)
            )
    
    total_time = time.time() - start_time
    
    return BatchOCRResponse(
        success=failed == 0,
        results=results,
        total_processing_time=total_time,
        total_images=len(images),
        successful=successful,
        failed=failed
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return prometheus_client.generate_latest()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "handwriting_api.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )