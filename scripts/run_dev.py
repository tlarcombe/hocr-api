#!/usr/bin/env python3
"""
Development server runner for Handwriting Recognition API.
This script sets up the development environment and runs the API with hot reload.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import signal
import time
from typing import Optional


def setup_environment():
    """Set up development environment variables."""
    env_vars = {
        'HANDWRITING_LOG_LEVEL': 'DEBUG',
        'HANDWRITING_DEBUG': 'true',
        'HANDWRITING_RELOAD': 'true',
        'HANDWRITING_OCR_USE_GPU': 'false',
        'HANDWRITING_HOST': '0.0.0.0',
        'HANDWRITING_PORT': '8000',
        'HANDWRITING_ENABLE_METRICS': 'true',
        'HANDWRITING_MAX_CONCURRENT_REQUESTS': '5',
        'PYTHONPATH': str(Path(__file__).parent.parent / 'src'),
    }
    
    for key, value in env_vars.items():
        os.environ.setdefault(key, value)


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        from PIL import Image
        print("✓ Core dependencies are installed")
        
        # Check for PaddleOCR, but don't fail if it's not available
        try:
            import paddleocr
            print("✓ PaddleOCR is available")
            return True
        except ImportError:
            print("⚠️  PaddleOCR not available, using minimal OCR engine")
            print("✓ Will use fallback minimal OCR engine for demonstration")
            return True
            
    except ImportError as e:
        print(f"✗ Missing core dependency: {e}")
        print("Please install: pip install fastapi uvicorn pillow")
        return False


def create_directories():
    """Create necessary directories for development."""
    directories = [
        'logs',
        'tmp',
        'tests/fixtures'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def run_tests():
    """Run a quick test to verify setup."""
    print("Running quick tests...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/test_api.py', '-v', '-k', 'test_health_check'],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        if result.returncode == 0:
            print("✓ Quick tests passed")
            return True
        else:
            print(f"✗ Tests failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Could not run tests: {e}")
        return False


def create_sample_image():
    """Create a sample image for testing."""
    try:
        from PIL import Image, ImageDraw
        import io
        
        # Create a simple test image with text-like patterns
        image = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw some simple patterns that might be recognized as text
        draw.rectangle([50, 50, 150, 100], outline='black', width=2)
        draw.rectangle([200, 50, 350, 100], outline='black', width=2)
        draw.line([50, 150, 350, 150], fill='black', width=3)
        
        # Save the image
        image_path = Path('tests/fixtures/sample_handwriting.jpg')
        image.save(image_path, 'JPEG', quality=90)
        print(f"✓ Created sample image: {image_path}")
        return True
        
    except ImportError:
        print("✗ PIL not available, skipping sample image creation")
        return False
    except Exception as e:
        print(f"✗ Could not create sample image: {e}")
        return False


def run_server(host: str = '0.0.0.0', port: int = 8000, reload: bool = True):
    """Run the development server."""
    print(f"Starting development server on {host}:{port}")
    print(f"API documentation will be available at: http://{host}:{port}/docs")
    print(f"Health check: http://{host}:{port}/health")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'handwriting_api.api.main:app',
            '--host', host,
            '--port', str(port),
            '--log-level', 'debug'
        ]
        
        if reload:
            cmd.append('--reload')
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n✓ Server stopped by user")
    except Exception as e:
        print(f"✗ Server error: {e}")
        sys.exit(1)


def test_api_endpoint(base_url: str = 'http://localhost:8000'):
    """Test basic API functionality."""
    import requests
    import time
    
    print("Testing API endpoints...")
    
    try:
        # Test health endpoint
        print(f"Testing health endpoint: {base_url}/health")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            print("✓ Health check passed")
            health_data = response.json()
            print(f"  Status: {health_data['status']}")
            print(f"  Version: {health_data['version']}")
            print(f"  Model loaded: {health_data['model_loaded']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
            
        # Test model info endpoint
        print(f"Testing model info: {base_url}/model-info")
        response = requests.get(f"{base_url}/model-info", timeout=10)
        
        if response.status_code == 200:
            print("✓ Model info endpoint working")
            model_data = response.json()
            print(f"  Engine: {model_data['engine']}")
            print(f"  Language: {model_data['language']}")
        else:
            print(f"✗ Model info failed: {response.status_code}")
            return False
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ API server not responding")
        return False
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run Handwriting Recognition API development server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--no-reload', action='store_true', help='Disable auto-reload')
    parser.add_argument('--skip-tests', action='store_true', help='Skip initial tests')
    parser.add_argument('--skip-sample-image', action='store_true', help='Skip sample image creation')
    parser.add_argument('--test-only', action='store_true', help='Only run tests, then exit')
    
    args = parser.parse_args()
    
    print("🚀 Handwriting Recognition API Development Server")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create sample image
    if not args.skip_sample_image:
        create_sample_image()
    
    # Run tests
    if not args.skip_tests:
        print("\nRunning initial tests...")
        if not run_tests():
            print("\n⚠️  Some tests failed, but continuing with server startup...")
    
    # Test-only mode
    if args.test_only:
        print("\nTest-only mode completed.")
        return
    
    # Test existing API if it's already running
    print("\nTesting existing API...")
    if test_api_endpoint(f"http://{args.host}:{args.port}"):
        print("✓ API is already running and responding")
        print(f"Visit: http://{args.host}:{args.port}/docs for documentation")
        return
    
    # Run the server
    print("\nStarting server...")
    run_server(args.host, args.port, reload=not args.no_reload)


if __name__ == '__main__':
    main()