"""Application settings and configuration."""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    app_name: str = Field(default="Handwriting Recognition API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload")
    
    # OCR Configuration
    ocr_lang: str = Field(default="en", description="OCR language code")
    ocr_use_angle_cls: bool = Field(default=True, description="Use angle classification")
    ocr_use_gpu: bool = Field(default=True, description="Use GPU for OCR processing")
    ocr_det_model_dir: Optional[str] = Field(default=None, description="Detection model directory")
    ocr_rec_model_dir: Optional[str] = Field(default=None, description="Recognition model directory")
    ocr_cls_model_dir: Optional[str] = Field(default=None, description="Classification model directory")
    
    # Image Processing Configuration
    max_image_size: int = Field(default=10 * 1024 * 1024, description="Maximum image size in bytes (10MB)")
    allowed_image_types: List[str] = Field(
        default=["image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic"],
        description="Allowed image MIME types"
    )
    max_image_dimensions: tuple = Field(default=(4096, 4096), description="Maximum image dimensions (width, height)")
    
    # Processing Configuration
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent OCR requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    
    # Security Configuration
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    max_request_size: int = Field(default=50 * 1024 * 1024, description="Maximum request size in bytes (50MB)")
    
    # Model Configuration
    model_cache_size: int = Field(default=1, description="Number of OCR models to cache")
    enable_model_warmup: bool = Field(default=True, description="Preload OCR model on startup")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variables prefix
        env_prefix = "HANDWRITING_"


# Global settings instance
settings = Settings()