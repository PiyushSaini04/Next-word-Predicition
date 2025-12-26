# Project Summary: Next Word Prediction (Hinglish)

## Overview
This project focuses on building a **Next Word Prediction** model for **Hinglish** (Hindi + English) conversational text. The goal is to train a Transformer-based Causal Language Model (approx. 5M parameters) on a 1-million-line dataset.

## Key Components
1. **Dataset**: ~1 million lines of conversational pairs, merged into a continuous training sequence.
2. **Tokenizer**: SentencePiece (BPE) with a vocab size of **4000**.
3. **Model**: Custom Transformer (PyTorch).
    - Embedding Dim: 256
    - Layers: 6
    - Heads: 8
    - Block Size/Context: 128
4. **Training**:
    - Optimizer: AdamW
    - Scheduler: Cosine Annealing
    - Mixed Precision (AMP) enabled
    - Label Smoothing: 0.1
5. **Deployment**:
    - Backend: Python (Flask/FastAPI)
    - Frontend: Simple HTML/JS interface

## Current Status
- [x] Data Cleaning & Preprocessing
- [x] Tokenizer Training
- [x] Model Architecture Implementation
- [x] Training Script Implementation
- [ ] Model Training (Pending Execution)
- [ ] Deployment Setup

## Hardware Requirements
- **GPU**: Recommended (CUDA) for efficient training.
- **RAM**: ~8GB+ sufficient for this model size.
