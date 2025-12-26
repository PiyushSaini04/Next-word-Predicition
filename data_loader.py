"""
Data preparation module for Causal Language Modeling.
Concatenates all text and chunks it into block_size sequences for efficient training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import os

class CausalLMDataset(Dataset):
    """
    Dataset for Causal Language Modeling (Next Token Prediction).
    Chunks text into blocks of size `block_size` + 1.
    Input: block[:-1]
    Target: block[1:]
    """
    
    def __init__(
        self,
        token_ids: List[int],
        block_size: int = 64
    ):
        self.token_ids = token_ids
        self.block_size = block_size
        
        # Calculate number of blocks
        # We need block_size + 1 tokens for each sample (input + target)
        self.total_tokens = len(token_ids)
        self.num_samples = self.total_tokens // (block_size + 1)
        
        print(f"Dataset created with {self.num_samples} samples from {self.total_tokens} tokens")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get start index
        start_idx = idx * (self.block_size + 1)
        end_idx = start_idx + self.block_size + 1
        
        # Get chunk
        chunk = self.token_ids[start_idx:end_idx]
        
        # Convert to tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        
        # Split into input and target
        input_ids = chunk_tensor[:-1]
        target_ids = chunk_tensor[1:]
        
        return input_ids, target_ids


def create_dataloaders(
    text_file: str,
    tokenizer_path: str,
    block_size: int = 64,  # Replaces max_length
    batch_size: int = 32,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,  # Simple train/val split is usually enough for LM
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    """
    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    
    # Load and tokenize all text
    print(f"Loading and tokenizing {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        # Read all lines
        lines = [line.strip() for line in f if line.strip()]
    
    full_token_ids = []
    # Add BOS/EOS tokens if needed, but for continuous LM usually just concat with separator
    # SentencePiece handles text well. Let's just encode each line and concat.
    for line in lines:
        ids = tokenizer.encode(line, out_type=int)
        full_token_ids.extend(ids)
        full_token_ids.append(tokenizer.eos_id()) # Add EOS between sentences
        
    print(f"Total tokens in corpus: {len(full_token_ids)}")
    
    # Split tokens directly (simple split)
    # Be careful not to split in middle of a sentence? 
    # For large corpora, random split of lines is better, then tokenizing.
    
    # Let's split LINES first to avoid data leakage
    train_lines, val_lines = train_test_split(lines, test_size=val_ratio, random_state=seed)
    
    def encode_lines(lines_list):
        ids = []
        for line in lines_list:
            ids.extend(tokenizer.encode(line, out_type=int))
            ids.append(tokenizer.eos_id())
        return ids
        
    train_ids = encode_lines(train_lines)
    val_ids = encode_lines(val_lines)
    
    print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")
    
    # Create Datasets
    train_dataset = CausalLMDataset(train_ids, block_size)
    val_dataset = CausalLMDataset(val_ids, block_size)
    
    # Create DataLoaders
    # num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader
