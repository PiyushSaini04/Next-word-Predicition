# Quickstart Guide

follow these steps to reproduce the project from scratch.

## 1. Setup Environment
Ensure you have Python installed. Install dependencies:
```bash
pip install torch torchvision torchaudio sentencepiece flask
# Optional for plotting
pip install matplotlib
```

## 2. Prepare Data
Place your `hinglish_conversations.csv` in the root directory.
Then run:
```bash
python preprocess_data.py
# Output: data/cleaned_hinglish.txt
```

## 3. Train Tokenizer
```bash
python train_tokenizer.py
# Output: models/tokenizer.model, models/tokenizer.vocab
```

## 4. Train Model
```bash
python train.py
# Monitoring: Checks loss every 100 steps.
# Output: models/best_model.pth, models/training_plot.png
```

## 5. Inference (Test Model)
After training, run:
```bash
python inference.py
```
Type a sentence to see predictions.

## 6. Run Web Demo
Start the backend:
```bash
python backend.py
```
Open `frontend/index.html` in your browser to interact.
