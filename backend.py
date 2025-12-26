from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import sentencepiece as spm
from model import TransformerLanguageModel, ModelConfig
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/best_model.pth"
TOKENIZER_PATH = "models/tokenizer.model"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = None
sp = None

def load_resources():
    global model, sp
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("Warning: Model or Tokenizer not found. Prediction will fail.")
        return

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    vocab_size = sp.get_piece_size()

    # Initialize model config
    config = ModelConfig(vocab_size=vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=128)
    model = TransformerLanguageModel(config)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

        # Resize embeddings if needed
        old_wpe = state_dict['transformer.wpe.weight']
        if old_wpe.shape[0] != vocab_size:
            print(f"Resizing embeddings from {old_wpe.shape[0]} to {vocab_size}")
            new_wpe = model.transformer.wpe.weight.data
            n_copy = min(old_wpe.shape[0], vocab_size)
            new_wpe[:n_copy] = old_wpe[:n_copy]
            state_dict['transformer.wpe.weight'] = new_wpe

        # Remove attention biases to avoid mismatch
        for i in range(config.n_layer):
            attn_bias_key = f'transformer.h.{i}.attn.bias'
            if attn_bias_key in state_dict:
                print(f"Removing mismatched key: {attn_bias_key}")
                del state_dict[attn_bias_key]

        # Load weights safely
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully with compatible weights.")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or sp is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    text = data.get('text', '')

    # Tokenize input
    ids = sp.encode(text)
    if len(ids) > model.config.block_size:
        ids = ids[-model.config.block_size:]

    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # Run model inference
    with torch.no_grad():
        logits, _ = model(x)

    last_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(last_logits, dim=-1)

    # Return top 5 predictions
    top_vals, top_idxs = torch.topk(probs, 5)
    suggestions = [{'word': sp.decode([idx.item()]), 'probability': float(val)} 
                   for val, idx in zip(top_vals, top_idxs)]

    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    load_resources()
    app.run(port=5000, debug=True)
