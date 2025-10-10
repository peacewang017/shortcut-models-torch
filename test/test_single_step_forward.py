import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model import DiT

def test_single_step_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    BATCH_SIZE = 1
    IMG_SIZE = 32
    PATCH_SIZE = 16
    IN_CHANNELS = 4
    HIDDEN_SIZE = 120
    NUM_HEADS = 12
    DIT_DEPTH = 12
    NUM_CLASSES = 2
    MLP_RATIO = 4.0
    DROPOUT_RATE = 0.0
    OUT_CHANNELS = IN_CHANNELS
    
    print(f"--- single step forward test on {device} ---")

    # config
    model = DiT(
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        dit_depth=DIT_DEPTH,
        mlp_ratio=MLP_RATIO,
        dropout_rate=DROPOUT_RATE,
        out_channels=OUT_CHANNELS
    ).to(device)
    
    model.eval()

    dummy_x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    dummy_t = torch.randint(0, 1000, (BATCH_SIZE,), device=device)
    dummy_dt = torch.randint(0, 1000, (BATCH_SIZE,), device=device)
    dummy_y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)
    
    print(f"input x shape: {dummy_x.shape}")
    print(f"input t shape: {dummy_t.shape}")
    print(f"input y shape: {dummy_y.shape}\n")

    # do forward
    print("forward:")
    with torch.no_grad():
        output = model(dummy_x, dummy_t, dummy_dt, dummy_y)
    print("forward finished!\n")
    
    print(f"output x shape: {output.shape}")
    
    # check shape
    expected_shape = (BATCH_SIZE, OUT_CHANNELS, IMG_SIZE, IMG_SIZE)
    
    assert output.shape == expected_shape, f"error shape, expect: {expected_shape}, get: {output.shape}"
    print("✅ successful forward")
    
if __name__ == '__main__':
    test_single_step_forward()