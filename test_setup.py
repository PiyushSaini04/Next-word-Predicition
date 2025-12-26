"""
Quick test script to verify the setup is correct.
Run this before starting training to check if everything is properly installed.
"""

import sys
import os

def check_imports():
    """Check if all required packages are installed."""
    print("Checking required packages...")
    
    packages = {
        'torch': 'PyTorch',
        'sentencepiece': 'SentencePiece',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed!")
    return True


def check_files():
    """Check if required files exist."""
    print("\nChecking project files...")
    
    required_files = [
        'preprocess_data.py',
        'train_tokenizer.py',
        'data_loader.py',
        'model.py',
        'train.py',
        'inference.py',
        'backend.py',
        'requirements.txt'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    print("\n✓ All required files present!")
    return True


def check_data():
    """Check if data files exist."""
    print("\nChecking data files...")
    
    csv_file = 'hinglish_english_parallel_corpus.csv'
    cleaned_file = 'data/cleaned_hinglish.txt'
    
    if os.path.exists(csv_file):
        print(f"  ✓ {csv_file} found")
    else:
        print(f"  ⚠ {csv_file} not found (you'll need to provide your CSV)")
    
    if os.path.exists(cleaned_file):
        print(f"  ✓ {cleaned_file} found (preprocessing already done)")
    else:
        print(f"  ⚠ {cleaned_file} not found (run preprocessing first)")
    
    return True


def check_models():
    """Check if model files exist."""
    print("\nChecking model files...")
    
    tokenizer = 'models/hinglish_tokenizer.model'
    model = 'models/hinglish_model.pth'
    
    if os.path.exists(tokenizer):
        print(f"  ✓ Tokenizer found")
    else:
        print(f"  ⚠ Tokenizer not found (run tokenizer training)")
    
    if os.path.exists(model):
        print(f"  ✓ Model found")
    else:
        print(f"  ⚠ Model not found (run model training)")
    
    return True


def check_cuda():
    """Check if CUDA is available."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠ CUDA not available - will use CPU (slower)")
    except ImportError:
        print("  ⚠ PyTorch not installed - cannot check CUDA")
    
    return True


def main():
    print("="*60)
    print("Hinglish Next-Word Prediction - Setup Check")
    print("="*60)
    
    all_ok = True
    
    all_ok &= check_imports()
    all_ok &= check_files()
    check_data()
    check_models()
    check_cuda()
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ Setup check complete! You're ready to start.")
        print("\nNext steps:")
        print("1. Run: python run_pipeline.py --csv hinglish_english_parallel_corpus.csv")
        print("2. Or follow the manual steps in QUICKSTART.md")
    else:
        print("⚠ Some issues found. Please fix them before proceeding.")
    print("="*60)


if __name__ == "__main__":
    main()



