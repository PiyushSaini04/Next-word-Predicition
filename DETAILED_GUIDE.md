# Detailed Project Guide

## 1. Data Pipeline
The data is sourced from `hinglish_conversations.csv`.
- **Preprocessing**: `preprocess_data.py` reads the CSV, cleans text (removes extra spaces/newlines), and merges input/output pairs into a single text stream for causal modeling.
- **Tokenization**: `train_tokenizer.py` trains a SentencePiece BPE model with vocab size 4000. This handles the mixed nature of Hinglish effectively.

## 2. Model Architecture
We use a **Decoder-only Transformer** (GPT-style).
Configuration:
- `vocab_size`: 4000
- `n_embd`: 256 (Embedding dimension)
- `n_head`: 8 (Attention heads)
- `n_layer`: 6 (Transformer blocks)
- `block_size`: 128 (Context window)

Total Parameters: ~5 Million. This is lightweight enough for quick training but capable of learning conversational patterns.

## 3. Training Process (`train.py`)
- **Dataset**: Custom `TextDataset` loads tokenized data.
- **Optimization**:
    - **AdamW**: Weight decay 0.1 for regularization.
    - **Cosine Scheduler**: Smooth learning rate decay.
    - **AMP (Automatic Mixed Precision)**: Speeds up training and reduces memory usage on NVIDIA GPUs.
    - **Gradient Clipping**: Prevents exploding gradients.
- **Loss**: CrossEntropy with Label Smoothing (0.1) to prettify distribution.

## 4. Inference Strategy
- **Sampling**: Top-k or Top-p (Nucleus) sampling can be used.
- **Greedy**: For next-word suggestion, we often just want the highest probability token.

## 5. Deployment
- **Backend**: A simple Python server (e.g., Flask) exposes a `/predict` endpoint.
- **Frontend**: A minimal HTML/CSS/JS page queries the backend and updates the UI in real-time.
