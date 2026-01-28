# Multi-stage Docker build for Handwriting Recognition API
FROM python:3.13-slim as builder

# Install system dependencies with stable package versions
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    ffmpeg \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libdc1394-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.13-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    ffmpeg \
    libpng16-16 \
    libjpeg62-turbo \
    libtiff6 \
    libdc1394-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create directories for logs and temporary files
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV HANDWRITING_LOG_LEVEL=INFO
ENV HANDWRITING_OCR_USE_GPU=false
ENV HANDWRITING_HOST=0.0.0.0
ENV HANDWRITING_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "handwriting_api.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]


# Development stage
FROM python:3.13-slim as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    ffmpeg \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libdc1394-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install in development mode
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-asyncio pytest-cov black isort flake8 mypy

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./
COPY tests/ ./tests/

# Install package in development mode
RUN pip install -e .

# Set environment variables for development
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV HANDWRITING_LOG_LEVEL=DEBUG
ENV HANDWRITING_DEBUG=true
ENV HANDWRITING_RELOAD=true

# Expose port
EXPOSE 8000

# Run development server with auto-reload
CMD ["python", "-m", "uvicorn", "handwriting_api.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]


# GPU-enabled stage (for systems with NVIDIA Docker runtime)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    ffmpeg \
    libpng16-16 \
    libjpeg62-turbo \
    libtiff6 \
    libdc1394-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt pyproject.toml ./

# Install Python dependencies with GPU support
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir paddlepaddle-gpu>=3.3.0

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create directories and set permissions
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables for GPU
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV HANDWRITING_LOG_LEVEL=INFO
ENV HANDWRITING_OCR_USE_GPU=true
ENV HANDWRITING_HOST=0.0.0.0
ENV HANDWRITING_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application with GPU support
CMD ["python", "-m", "uvicorn", "handwriting_api.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]