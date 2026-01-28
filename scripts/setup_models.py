#!/usr/bin/env python3
"""
Model setup script for Handwriting Recognition API.
Downloads and configures OCR models for different languages.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import urllib.request
import zipfile
import shutil
from typing import List, Dict


# Model configurations
MODEL_CONFIGS = {
    'en': {
        'name': 'English',
        'det_model': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
        'rec_model': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar',
        'cls_model': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
    },
    'ch': {
        'name': 'Chinese',
        'det_model': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar',
        'rec_model': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar',
        'cls_model': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
    },
    'multilingual': {
        'name': 'Multilingual',
        'det_model': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar',
        'rec_model': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_rec_infer.tar',
        'cls_model': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
    }
}


def download_file(url: str, destination: Path) -> bool:
    """Download a file with progress tracking."""
    try:
        print(f"Downloading {url}...")
        
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, reporthook=download_progress)
        print()  # New line after progress
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_tar(tar_path: Path, extract_to: Path) -> bool:
    """Extract tar file."""
    try:
        print(f"Extracting {tar_path}...")
        
        import tarfile
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_to)
        
        # Remove the tar file after extraction
        tar_path.unlink()
        return True
        
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False


def setup_language_models(language: str, models_dir: Path) -> bool:
    """Set up models for a specific language."""
    if language not in MODEL_CONFIGS:
        print(f"Language '{language}' not supported")
        return False
    
    config = MODEL_CONFIGS[language]
    lang_dir = models_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSetting up {config['name']} models...")
    
    # Download and extract detection model
    if config['det_model']:
        det_tar = lang_dir / 'det_model.tar'
        if download_file(config['det_model'], det_tar):
            extract_tar(det_tar, lang_dir)
    
    # Download and extract recognition model
    if config['rec_model']:
        rec_tar = lang_dir / 'rec_model.tar'
        if download_file(config['rec_model'], rec_tar):
            extract_tar(rec_tar, lang_dir)
    
    # Download and extract classification model
    if config['cls_model']:
        cls_tar = lang_dir / 'cls_model.tar'
        if download_file(config['cls_model'], cls_tar):
            extract_tar(cls_tar, lang_dir)
    
    print(f"✓ {config['name']} models setup complete")
    return True


def verify_models(models_dir: Path, language: str) -> bool:
    """Verify that models are properly set up."""
    lang_dir = models_dir / language
    
    if not lang_dir.exists():
        print(f"✗ Language directory not found: {lang_dir}")
        return False
    
    # Check for model files
    expected_files = [
        'inference.pdiparams',
        'inference.pdmodel',
        'inference.pdiparams.info'
    ]
    
    # Find model directories
    model_dirs = []
    for item in lang_dir.iterdir():
        if item.is_dir() and 'infer' in item.name:
            model_dirs.append(item)
    
    if not model_dirs:
        print(f"✗ No model directories found in {lang_dir}")
        return False
    
    all_valid = True
    for model_dir in model_dirs:
        print(f"Checking {model_dir.name}...")
        for expected_file in expected_files:
            file_path = model_dir / expected_file
            if file_path.exists():
                size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✓ {expected_file} ({size:.1f} MB)")
            else:
                print(f"  ✗ Missing {expected_file}")
                all_valid = False
    
    return all_valid


def list_available_languages():
    """List all available languages."""
    print("Available languages:")
    for lang_code, config in MODEL_CONFIGS.items():
        print(f"  {lang_code}: {config['name']}")


def get_model_size(models_dir: Path, language: str) -> Dict[str, float]:
    """Get the size of models for a language."""
    lang_dir = models_dir / language
    sizes = {}
    
    if not lang_dir.exists():
        return sizes
    
    for model_dir in lang_dir.iterdir():
        if model_dir.is_dir():
            total_size = 0
            for file in model_dir.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
            sizes[model_dir.name] = total_size / (1024 * 1024)  # MB
    
    return sizes


def cleanup_models(models_dir: Path, language: str = None):
    """Clean up downloaded models."""
    if language:
        # Clean up specific language
        lang_dir = models_dir / language
        if lang_dir.exists():
            shutil.rmtree(lang_dir)
            print(f"✓ Cleaned up {language} models")
    else:
        # Clean up all models
        if models_dir.exists():
            shutil.rmtree(models_dir)
            print("✓ Cleaned up all models")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup OCR models for Handwriting Recognition API')
    parser.add_argument('--language', '-l', default='en', 
                       help='Language to setup (default: en)')
    parser.add_argument('--models-dir', '-d', default='models',
                       help='Models directory (default: models)')
    parser.add_argument('--list-languages', action='store_true',
                       help='List available languages')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing models')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up models')
    parser.add_argument('--cleanup-all', action='store_true',
                       help='Clean up all models')
    parser.add_argument('--size', action='store_true',
                       help='Show model sizes')
    
    args = parser.parse_args()
    
    print("🏗️  Handwriting Recognition API - Model Setup")
    print("=" * 50)
    
    models_dir = Path(args.models_dir)
    
    # List languages
    if args.list_languages:
        list_available_languages()
        return
    
    # Cleanup
    if args.cleanup_all:
        cleanup_models(models_dir)
        return
    
    if args.cleanup:
        cleanup_models(models_dir, args.language)
        return
    
    # Verify existing models
    if args.verify:
        print(f"Verifying {args.language} models...")
        if verify_models(models_dir, args.language):
            print("✓ All models are valid")
        else:
            print("✗ Some models are missing or corrupted")
        return
    
    # Show model sizes
    if args.size:
        sizes = get_model_size(models_dir, args.language)
        if sizes:
            print(f"Model sizes for {args.language}:")
            total_size = 0
            for model_name, size in sizes.items():
                print(f"  {model_name}: {size:.1f} MB")
                total_size += size
            print(f"  Total: {total_size:.1f} MB")
        else:
            print(f"No models found for {args.language}")
        return
    
    # Setup models
    print(f"Setting up models for language: {args.language}")
    print(f"Models directory: {models_dir.absolute()}")
    
    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup language models
    if setup_language_models(args.language, models_dir):
        print("\n✓ Model setup completed successfully")
        
        # Verify models
        print("\nVerifying models...")
        if verify_models(models_dir, args.language):
            print("✓ All models are valid and ready to use")
            
            # Show sizes
            sizes = get_model_size(models_dir, args.language)
            if sizes:
                total_size = sum(sizes.values())
                print(f"✓ Total model size: {total_size:.1f} MB")
        else:
            print("✗ Model verification failed")
            sys.exit(1)
    else:
        print("✗ Model setup failed")
        sys.exit(1)
    
    print(f"\n🎉 Model setup complete!")
    print(f"You can now use the API with language: {args.language}")
    print(f"Set the environment variable: HANDWRITING_OCR_LANG={args.language}")


if __name__ == '__main__':
    main()