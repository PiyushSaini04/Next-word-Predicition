import sentencepiece as spm
import os

def train_tokenizer(input_file, model_prefix, vocab_size):
    print(f"Training tokenizer on {input_file}...")
    print(f"Vocab size: {vocab_size}")
    
    # Ensure raw data is treated as such, avoiding normalization if we already did it, 
    # but SentencePiece's default normalizer is usually good.
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0, # Use 1.0 to include all chars if possible, or 0.9995
        # 4000 vocab is small, so we might need to be careful with coverage if there are many rare chars.
        # But for hinglish (roman script mostly), it should be fine.
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]',
        user_defined_symbols=['[SEP]', '[MASK]']
    )
    print(f"Tokenizer trained. Saved as {model_prefix}.model and {model_prefix}.vocab")

if __name__ == "__main__":
    input_file = "data/cleaned_hinglish.txt"
    model_dir = "models"
    model_prefix = os.path.join(model_dir, "tokenizer")
    
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(input_file):
        train_tokenizer(input_file, model_prefix, vocab_size=4000)
    else:
        print(f"Error: {input_file} not found. Please run preprocess_data.py first.")
