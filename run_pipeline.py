"""
Complete pipeline script to run the entire project from preprocessing to training.
This script automates the entire workflow.
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Run complete Hinglish prediction pipeline")
    parser.add_argument("--csv", type=str, default="hinglish_conversations.csv",
                       help="Path to input CSV file")
    parser.add_argument("--text-column", type=str, default="Sentence",
                       help="Name of the column containing Hinglish text")
    parser.add_argument("--skip-preprocess", action="store_true",
                       help="Skip preprocessing if already done")
    parser.add_argument("--skip-tokenizer", action="store_true",
                       help="Skip tokenizer training if already done")
    parser.add_argument("--skip-train", action="store_true",
                       help="Skip model training if already done")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Preprocess data
    if not args.skip_preprocess:
        preprocess_cmd = (
            f'python preprocess_data.py '
            f'--csv "{args.csv}" '
            f'--output data/cleaned_hinglish.txt '
            f'--text-column "{args.text_column}" '
            f'--skip-rows 1'
        )
        run_command(preprocess_cmd, "Step 1: Preprocessing dataset")
    else:
        print("\n⏭️  Skipping preprocessing (--skip-preprocess flag set)")
    
    # Step 2: Train tokenizer
    if not args.skip_tokenizer:
        tokenizer_cmd = (
            'python train_tokenizer.py '
            '--input data/cleaned_hinglish.txt '
            '--model-prefix models/hinglish_tokenizer '
            '--vocab-size 8000'
        )
        run_command(tokenizer_cmd, "Step 2: Training SentencePiece tokenizer")
    else:
        print("\n⏭️  Skipping tokenizer training (--skip-tokenizer flag set)")
    
    # Step 3: Train model
    if not args.skip_train:
        train_cmd = (
            f'python train.py '
            f'--text-file data/cleaned_hinglish.txt '
            f'--tokenizer models/hinglish_tokenizer.model '
            f'--output-dir models '
            f'--epochs {args.epochs} '
            f'--batch-size {args.batch_size}'
        )
        run_command(train_cmd, "Step 3: Training Transformer model")
    else:
        print("\n⏭️  Skipping model training (--skip-train flag set)")
    
    print("\n" + "="*60)
    print("🎉 Pipeline completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the backend server: python backend.py")
    print("2. Open static/index.html in your browser")
    print("3. Start typing in Hinglish and see predictions!")


if __name__ == "__main__":
    main()



