import torch
import sentencepiece as spm
from model import TransformerLanguageModel, ModelConfig
import os

# Configuration
MODEL_PATH = "models/best_model.pth"
TOKENIZER_PATH = "models/tokenizer.model"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. First run train.py")
        
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    vocab_size = sp.get_piece_size()

    config = ModelConfig(vocab_size=vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=128)
    model = TransformerLanguageModel(config)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Resize embeddings if needed
    old_wpe = state_dict['transformer.wpe.weight']
    if old_wpe.shape[0] != vocab_size:
        print(f"Resizing embeddings from {old_wpe.shape[0]} to {vocab_size}")
        new_wpe = model.transformer.wpe.weight.data
        n_copy = min(old_wpe.shape[0], vocab_size)
        new_wpe[:n_copy] = old_wpe[:n_copy]
        state_dict['transformer.wpe.weight'] = new_wpe

    # Remove attention biases from state_dict to avoid shape mismatch
    for i in range(config.n_layer):
        attn_bias_key = f'transformer.h.{i}.attn.bias'
        if attn_bias_key in state_dict:
            print(f"Removing mismatched key: {attn_bias_key}")
            del state_dict[attn_bias_key]

    # Load remaining weights
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully with compatible weights.")
    return model, sp

def predict_next_word(model, sp, text, top_k=5):
    ids = sp.encode(text)
    if len(ids) > model.config.block_size:
        ids = ids[-model.config.block_size:]
        
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        logits, _ = model(x)
    
    last_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(last_logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probs, top_k)
    results = [(sp.decode([idx.item()]), val.item()) for val, idx in zip(top_probs, top_indices)]
    return results

if __name__ == "__main__":
    try:
        model, sp = load_model()
        print("Type a sentence to predict the next word (Ctrl+C to exit).")
        
        while True:
            text = input("\nInput: ")
            if not text: 
                continue
            
            suggestions = predict_next_word(model, sp, text)
            print("Suggestions:")
            for word, prob in suggestions:
                print(f"  {word:<15} ({prob:.2%})")
                
    except FileNotFoundError as e:
        print(str(e))
    except KeyboardInterrupt:
        print("\nExiting...")

