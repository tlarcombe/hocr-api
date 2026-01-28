# HORC API - Handwriting Recognition API

A production-ready, AI-powered handwriting recognition API built with FastAPI and EasyOCR. Convert handwritten text from images (JPEG, PNG, WebP, HEIC) into editable digital text with high accuracy.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- SSH access to GitHub (already configured)

### Installation

```bash
# Clone the repository
git clone git@github.com:tlarcombe/horc-api.git
cd horc-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
PYTHONPATH=src python -m uvicorn handwriting_api.api.main:app --host 127.0.0.1 --port 8000
```

### Test the API

```bash
# Health check
curl http://127.0.0.1:8000/health

# Recognize handwriting from image
curl -X POST "http://127.0.0.1:8000/recognize" \
  -F "image=@your_handwriting.jpg" \
  -F "return_confidence=true"
```

## 📋 Features

- **Real OCR**: Uses EasyOCR for actual handwriting recognition (not mock data)
- **Multiple Formats**: Supports JPEG, PNG, WebP, HEIC image formats
- **Batch Processing**: Process multiple images in a single request
- **Confidence Scores**: Get confidence levels for each detected text region
- **Image Preprocessing**: Automatic image enhancement for better OCR results
- **RESTful API**: Clean, well-documented API with OpenAPI/Swagger documentation
- **Production Ready**: Docker containerization, monitoring, rate limiting
- **Comprehensive Testing**: Full test suite with pytest
- **Multilingual Support**: Support for 80+ languages via EasyOCR

## 🎯 API Endpoints

### 1. Health Check
**GET** `/health`

Returns API health status and system information.

### 2. Handwriting Recognition
**POST** `/recognize`

Upload an image and get recognized handwritten text.

**Parameters:**
- `image` (file, required): Image file (JPEG, PNG, WebP, HEIC)
- `return_confidence` (boolean, optional): Include confidence scores (default: true)
- `preprocess` (boolean, optional): Apply image preprocessing (default: true)

### 3. Batch Recognition
**POST** `/recognize/batch`

Process multiple images in a single request.

### 4. Model Information
**GET** `/model-info`

Get OCR model information and capabilities.

### 5. Metrics (Prometheus)
**GET** `/metrics`

Prometheus metrics endpoint for monitoring.

## 📊 Example Results

**Input Image**: Handwritten text on paper

**Output**:
```json
{
  "success": true,
  "text": "Shs\n&\nVRelvtins bxl\nTex\nkumi -2\nena led\nCa/a  .\nZ\nhaseoiky % ~Use noc dy",
  "results": [
    {
      "text": "Shs",
      "confidence": 0.2087294846674522,
      "coordinates": [[138.0,110.0],[429.0,110.0],[429.0,308.0],[138.0,308.0]]
    },
    {
      "text": "VRelvtins bxl",
      "confidence": 0.09626295677076693,
      "coordinates": [[480.0,76.0],[1391.0,76.0],[1391.0,310.0],[480.0,310.0]]
    }
  ],
  "processing_time": 21.5,
  "model_info": {
    "engine": "EasyOCR",
    "language": "en",
    "use_gpu": false,
    "model_version": "1.7+"
  }
}
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the API
curl http://localhost:8000/health
```

## 📈 Performance

- **Processing Time**: ~20-30 seconds per image (depends on complexity)
- **Supported Formats**: JPEG, PNG, WebP, HEIC
- **Max Image Size**: 10MB (configurable)
- **Max Dimensions**: 4096x4096 pixels (configurable)
- **Languages**: 80+ languages supported

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# API Configuration
HANDWRITING_LOG_LEVEL=INFO
HANDWRITING_HOST=0.0.0.0
HANDWRITING_PORT=8000

# OCR Configuration
HANDWRITING_OCR_LANG=en
HANDWRITING_OCR_USE_GPU=false

# Performance Settings
HANDWRITING_MAX_IMAGE_SIZE=10485760
HANDWRITING_MAX_CONCURRENT_REQUESTS=10
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_api.py -v
```

## 🚀 Production Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions including:
- Docker deployment
- Kubernetes setup
- Cloud deployment (AWS, GCP, Azure)
- Monitoring and scaling
- SSL/TLS configuration

## 📚 Documentation

- **API Documentation**: http://localhost:8000/docs (when debug mode enabled)
- **Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Research Summary**: [handwriting_recognition_research.md](handwriting_recognition_research.md)

## 🔍 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │───▶│   FastAPI       │───▶│   EasyOCR       │
│   (Web/Mobile)  │    │   REST API      │    │   OCR Engine    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Monitoring    │
                       │   (Prometheus)  │
                       └─────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for the excellent OCR engine
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for research and inspiration

---

**⭐ Star this repository if you find it useful!**