# Next-Word Prediction for Hinglish (Hindi-English Mixed Text)

A complete end-to-end system for next-word prediction in Hinglish (Hindi written in Roman letters) using a Transformer-based model. This system can be used in assistive keyboards for higher education students, providing real-time word suggestions similar to VSCode autocomplete.

## Features

- 🧹 **Data Preprocessing**: Cleans Hinglish text by removing mentions, URLs, hashtags, emojis, and special characters
- 🔤 **Custom Tokenizer**: SentencePiece BPE tokenizer trained specifically on Hinglish text (vocab size: 8000-10000)
- 🤖 **Transformer Model**: GPT-style decoder architecture with multi-head self-attention
- 🚀 **FastAPI Backend**: RESTful API for real-time predictions
- 💻 **Interactive Frontend**: VSCode-style autocomplete interface with Tab completion and click-to-insert
- 📱 **Responsive Design**: Works on both desktop and mobile devices

## Project Structure

```
.
├── preprocess_data.py          # Data cleaning and preprocessing
├── train_tokenizer.py           # SentencePiece tokenizer training
├── data_loader.py              # PyTorch Dataset and DataLoader
├── model.py                    # Transformer model architecture
├── train.py                    # Model training script
├── inference.py                # Inference module for predictions
├── backend.py                  # FastAPI server
├── run_pipeline.py             # Complete pipeline automation
├── static/                     # Frontend files
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/                       # Processed data (created during preprocessing)
├── models/                     # Trained models (created during training)
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start (Automated Pipeline)

Run the complete pipeline from preprocessing to training:

```bash
python run_pipeline.py --csv hinglish_english_parallel_corpus.csv --text-column Sentence
```

This will:
1. Preprocess the CSV dataset
2. Train the SentencePiece tokenizer
3. Train the Transformer model

### Manual Step-by-Step

#### Step 1: Preprocess Data

Clean the Hinglish text from your CSV file:

```bash
python preprocess_data.py --csv hinglish_english_parallel_corpus.csv --output data/cleaned_hinglish.txt --text-column Sentence --skip-rows 1
```

**Parameters:**
- `--csv`: Path to your CSV file
- `--output`: Output path for cleaned text
- `--text-column`: Column name containing Hinglish text (default: "Sentence")
- `--skip-rows`: Number of header rows to skip (default: 1)
- `--encoding`: File encoding if UTF-8 fails (try: latin-1, cp1252)

#### Step 2: Train Tokenizer

Train a SentencePiece tokenizer on the cleaned text:

```bash
python train_tokenizer.py --input data/cleaned_hinglish.txt --model-prefix models/hinglish_tokenizer --vocab-size 8000
```

**Parameters:**
- `--input`: Path to cleaned text file
- `--model-prefix`: Output prefix for tokenizer files
- `--vocab-size`: Vocabulary size (8000-10000)

#### Step 3: Train Model

Train the Transformer model:

```bash
python train.py --text-file data/cleaned_hinglish.txt --tokenizer models/hinglish_tokenizer.model --output-dir models --epochs 15 --batch-size 32
```

**Parameters:**
- `--text-file`: Path to cleaned text file
- `--tokenizer`: Path to tokenizer model
- `--output-dir`: Directory to save trained model
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size (default: 32)
- `--d-model`: Model dimension (default: 256)
- `--nhead`: Number of attention heads (default: 4)
- `--num-layers`: Number of transformer layers (default: 4)

#### Step 4: Start Backend Server

Start the FastAPI server:

```bash
python backend.py
```

The server will run on `http://localhost:8000` by default.

You can also set environment variables:
- `MODEL_PATH`: Path to model checkpoint (default: `models/hinglish_model.pth`)
- `TOKENIZER_PATH`: Path to tokenizer (default: `models/hinglish_tokenizer.model`)
- `PORT`: Server port (default: 8000)

#### Step 5: Open Frontend

Open `static/index.html` in your web browser. The frontend will connect to the backend API and provide real-time predictions as you type.

## API Usage

### Predict Next Words

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "main kal",
  "top_k": 5,
  "temperature": 1.0
}
```

**Response:**
```json
{
  "predictions": ["jaunga", "jaati", "jaaunga", "college", "kaam"]
}
```

**Parameters:**
- `text`: Input Hinglish text
- `top_k`: Number of predictions to return (default: 5)
- `temperature`: Sampling temperature - higher values = more diverse predictions (default: 1.0)

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Frontend Features

- **Real-time Predictions**: As you type, predictions appear automatically
- **Tab Completion**: Press `Tab` to accept the first suggestion
- **Click to Insert**: Click any suggestion to insert it
- **Arrow Key Navigation**: Use arrow keys to navigate through suggestions (Enter to accept)
- **Clean UI**: Modern, responsive design

## Model Architecture

The model uses a GPT-style Transformer decoder with:

- **Token Embeddings**: Learnable embeddings for each token
- **Positional Encoding**: Sinusoidal positional encodings
- **Multi-Head Self-Attention**: 4 attention heads with causal masking
- **Feed-Forward Networks**: GELU activation with dropout
- **Layer Normalization**: Applied after attention and FFN
- **Output Projection**: Linear layer to vocabulary size

**Default Hyperparameters:**
- Model dimension: 256
- Attention heads: 4
- Transformer layers: 4
- Feed-forward dimension: 1024
- Dropout: 0.1
- Max sequence length: 40 tokens
- Vocabulary size: 8000

## Dataset Format

The CSV file should have at least one column containing Hinglish text. Example:

```csv
Sentence,English_Translation
main kal college jaunga,I will go to college tomorrow
tum kahan ho,where are you
```

The preprocessing script will:
- Remove mentions (@username), URLs, hashtags, emojis
- Convert to lowercase
- Collapse multiple spaces
- Remove special characters

## Training Details

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW with learning rate 3e-4
- **Gradient Clipping**: Max norm 1.0
- **Validation Split**: 10% of data
- **Test Split**: 10% of data
- **Training Split**: 80% of data

## Testing Inference

Test the model directly from command line:

```bash
python inference.py --model models/hinglish_model.pth --tokenizer models/hinglish_tokenizer.model --text "main kal" --top-k 5
```

## Troubleshooting

### Encoding Issues

If you get encoding errors when reading the CSV:
```bash
python preprocess_data.py --csv your_file.csv --encoding latin-1
```

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py ... --batch-size 16
```

### Model Not Found

Make sure you've completed all training steps before starting the backend:
1. Preprocess data
2. Train tokenizer
3. Train model

## Performance Tips

- Use GPU for faster training: The model automatically uses CUDA if available
- Adjust batch size based on available memory
- Reduce `max_length` if sequences are shorter
- Increase `vocab_size` for better coverage (but slower training)

## Future Enhancements

- [ ] Multi-word suggestions (phrase completion)
- [ ] Spelling normalization for transliteration variations
- [ ] Confidence scores for predictions
- [ ] ONNX export for mobile deployment
- [ ] React/Flutter keyboard app
- [ ] User history-based suggestions

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this project, please cite:

```
Next-Word Prediction for Assistive Keyboards (Indic/Hinglish) for Higher Education
Transformer-based Hinglish Text Prediction System
```

## Contact

For questions or issues, please open an issue on the repository.



#   N e x t - w o r d - P r e d i c i t i o n  
 