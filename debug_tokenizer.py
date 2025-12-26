import sentencepiece as spm
import sys

tokenizer = spm.SentencePieceProcessor()
try:
    tokenizer.load("models/hinglish_tokenizer.model")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

texts = ["main kal", "tum", "college", "abcdefg"]

print(f"Vocab size: {tokenizer.get_piece_size()}")
print(f"PAD: {tokenizer.pad_id()}")
print(f"UNK: {tokenizer.unk_id()}")

for text in texts:
    ids = tokenizer.encode(text, out_type=int)
    pieces = tokenizer.encode(text, out_type=str)
    print(f"\nText: '{text}'")
    print(f"IDs: {ids}")
    print(f"Pieces: {pieces}")
