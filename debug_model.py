import torch
import sys
from model import NextWordPredictor

try:
    checkpoint = torch.load("models/hinglish_model.pth", map_location='cpu')
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

model = NextWordPredictor(
    vocab_size=checkpoint['vocab_size'],
    d_model=checkpoint['d_model'],
    nhead=checkpoint['nhead'],
    num_layers=checkpoint['num_layers'],
    dim_feedforward=checkpoint['dim_feedforward'],
    max_seq_length=checkpoint['max_length'],
    dropout=checkpoint['dropout'],
    pad_token_id=checkpoint['pad_token_id']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Model loaded.")
print(f"Vocab size: {model.token_embedding.num_embeddings}")

# Check weights
print("\nChecking weights:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
        # Only print first few to avoid spam
        if "layers.0" in name: break

# Test sensitivity
print("\nTesting Sensitivity:")
input1 = torch.LongTensor([[100, 200]]) # Arbitrary tokens
input2 = torch.LongTensor([[500, 600]]) # Different tokens

with torch.no_grad():
    out1 = model(input1)
    out2 = model(input2)

diff = (out1 - out2).abs().sum().item()
print(f"Difference between outputs for different inputs: {diff}")

if diff < 1e-6:
    print("CRITICAL: Model output is identical for different inputs!")
else:
    print("Model output changes with input.")
