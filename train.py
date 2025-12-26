# train.py (optimized)
import os
import time
import math
import json
import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import matplotlib.pyplot as plt
from model import TransformerLanguageModel, ModelConfig

# --- Configuration ---
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
MAX_EPOCHS = 15
EVAL_INTERVAL = 1
SAVE_DIR = "models"
DATA_FILE = "data/cleaned_hinglish.txt"
TOKENIZER_MODEL = "models/tokenizer.model"
PREPARED_DIR = "data/prepared"
TRAIN_BIN = os.path.join(PREPARED_DIR, "train.bin")
VAL_BIN = os.path.join(PREPARED_DIR, "val.bin")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Controls to speed up iteration
MAX_STEPS_PER_EPOCH = 2000   # set to None to process full epoch
SAMPLE_RATE = 1.0            # 1.0 = use all lines; <1.0 will randomly downsample lines while preparing
VAL_SPLIT = 0.01             # fraction for validation during prepare
BLOCK_SIZE = 64              # model context length (kept small to speed up)

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# --- Prepare tokenized binary files (line-by-line streaming) ---
def prepare_bin_files(txt_path, tokenizer_path, out_dir=PREPARED_DIR, sample_rate=SAMPLE_RATE, val_split=VAL_SPLIT):
    os.makedirs(out_dir, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    print(f"[prepare] Using tokenizer: {tokenizer_path}")

    # If bin files already exist, skip
    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        print("[prepare] Found existing .bin files — skipping prepare.")
        return

    # Tokenize line-by-line to avoid huge memory spikes and write tokens to temp numpy list segments
    token_chunks = []
    total_tokens = 0
    tmp_parts = []
    part_idx = 0
    lines_processed = 0

    with open(txt_path, 'r', encoding='utf-8') as f:
        buffer_ids = []
        for line in f:
            if sample_rate < 1.0 and np.random.rand() > sample_rate:
                continue
            line = line.strip()
            if not line:
                continue
            # encode line, append EOS token if available (use sp.eos_id() or just leave)
            ids = sp.encode(line)
            # optionally append EOS token if tokenizer defines one (this keeps examples separate)
            # append a single separator token (e.g., id 0) only if desired. Here we avoid inventing tokens.
            buffer_ids.extend(ids)
            lines_processed += 1

            # flush in chunks to keep memory low
            if len(buffer_ids) > 2_000_000:
                part_fn = os.path.join(out_dir, f"tmp_part_{part_idx}.npy")
                np.save(part_fn, np.array(buffer_ids, dtype=np.int32))
                tmp_parts.append(part_fn)
                part_idx += 1
                total_tokens += len(buffer_ids)
                buffer_ids = []

        # final buffer
        if buffer_ids:
            part_fn = os.path.join(out_dir, f"tmp_part_{part_idx}.npy")
            np.save(part_fn, np.array(buffer_ids, dtype=np.int32))
            tmp_parts.append(part_fn)
            total_tokens += len(buffer_ids)

    print(f"[prepare] Tokenized {lines_processed} lines into ~{total_tokens} tokens across {len(tmp_parts)} parts")

    # Concatenate parts to one big array (but do it in a memory-friendly way)
    # Count total length
    total_len = 0
    for p in tmp_parts:
        arr = np.load(p, mmap_mode='r')
        total_len += arr.shape[0]

    # Load concatenated into memory if it fits, else stream-split directly into train/val files
    all_tokens = np.empty(total_len, dtype=np.int32)
    offset = 0
    for p in tmp_parts:
        arr = np.load(p)
        L = arr.shape[0]
        all_tokens[offset:offset+L] = arr
        offset += L
        os.remove(p)  # cleanup temp part

    # split train/val
    nval = max(1, int(total_len * val_split))
    ntrain = total_len - nval
    train = all_tokens[:ntrain]
    val = all_tokens[ntrain:]

    # Save binary files (fast to load next time)
    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)
    print(f"[prepare] Wrote train.bin ({ntrain} tokens) and val.bin ({nval} tokens)")

# --- Loading token bins or tokenizing if needed ---
def load_token_ids(txt_path, tokenizer_path):
    # Prefer preprocessed .bin files
    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        print("[load] Loading tokens from binary files")
        train_tokens = np.fromfile(TRAIN_BIN, dtype=np.int32).tolist()
        val_tokens = np.fromfile(VAL_BIN, dtype=np.int32).tolist()
        vocab_size = spm.SentencePieceProcessor(model_file=tokenizer_path).get_piece_size()
        return train_tokens, val_tokens, vocab_size

    # else prepare and then load
    prepare_bin_files(txt_path, tokenizer_path)
    return load_token_ids(txt_path, tokenizer_path)

# --- Training ---
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("Data file not found. Run preprocess_data.py first.")
        return

    print("Preparing / loading tokenized data (this may take a while the first run)...")
    train_ids, val_ids, vocab_size = load_token_ids(DATA_FILE, TOKENIZER_MODEL)

    print(f"[train] Train tokens: {len(train_ids)} | Val tokens: {len(val_ids)} | Vocab size: {vocab_size}")

    config = ModelConfig(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=BLOCK_SIZE
    )

    train_dataset = TextDataset(train_ids, config.block_size)
    val_dataset = TextDataset(val_ids, config.block_size)

    # DataLoader settings: safe num_workers
    safe_workers = 0
    try:
        cpu_count = multiprocessing.cpu_count()
        # Use workers only if not windows and more than 1 CPU
        if os.name != 'nt' and cpu_count > 1:
            safe_workers = min(4, cpu_count // 2)
    except Exception:
        safe_workers = 0

    pin_memory = True if DEVICE == 'cuda' else False

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory, num_workers=safe_workers, persistent_workers=(safe_workers>0))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory, num_workers=safe_workers, persistent_workers=(safe_workers>0))

    print(f"[train] DataLoaders ready. train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model = TransformerLanguageModel(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-1)

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None

    history = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}
    best_val_loss = float('inf')

    print("Starting training...")
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits, loss = model(x, targets=y)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(x, targets=y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # scheduler step per update (ok since T_max set to total_steps)
            try:
                scheduler.step()
            except Exception:
                pass

            total_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Step {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

            # Fast debug bound: stop early in epoch if desired
            if MAX_STEPS_PER_EPOCH and (i+1) >= MAX_STEPS_PER_EPOCH:
                print(f"[train] Reached MAX_STEPS_PER_EPOCH ({MAX_STEPS_PER_EPOCH}) — breaking epoch early.")
                break

        avg_train_loss = total_loss / (i+1)
        history['train_loss'].append(avg_train_loss)

        # Validation
        if (epoch + 1) % EVAL_INTERVAL == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    if scaler:
                        with torch.cuda.amp.autocast():
                            _, loss = model(x, targets=y)
                    else:
                        _, loss = model(x, targets=y)
                    val_loss += loss.item()
                    val_steps += 1
                    # small val budget to speed iteration if desired:
                    if val_steps >= 200:
                        break

            avg_val_loss = val_loss / max(1, val_steps)
            perplexity = math.exp(avg_val_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_perplexity'].append(perplexity)

            print(f"--> Epoch {epoch+1} Completed. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
                print(f"[train] New best model saved (val_loss={best_val_loss:.4f})")

            # Save history each eval
            with open(os.path.join(SAVE_DIR, "history.json"), 'w') as f:
                json.dump(history, f)

    total_min = (time.time() - start_time) / 60.0
    print(f"Training complete in {total_min:.2f} minutes.")

    # Plot and save training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss' if history['val_loss'] else [])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_perplexity'], label='Val Perplexity' if history['val_perplexity'] else [])
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_plot.png"))
    print("[train] Training plot saved.")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
