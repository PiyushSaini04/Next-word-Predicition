# Quick Reference

| Script | Purpose | Output |
| :--- | :--- | :--- |
| `preprocess_data.py` | Converts CSV to clean TXT | `data/cleaned_hinglish.txt` |
| `train_tokenizer.py` | Trains SentencePiece Model | `models/tokenizer.model`, `models/tokenizer.vocab` |
| `model.py` | Contains Transformer Class | N/A (Imported module) |
| `train.py` | Trains the model | `models/best_model.pth`, `history.json` |
| `inference.py` | CLI for testing generation | Terminal output |
| `backend.py` | API Server for deplyment | Localhost server |

## Hyperparameters
- Max Epochs: 20
- Batch Size: 32
- Learning Rate: 5e-4
- Block Size: 128
- Vocab Size: 4000
