import torch
import numpy as np
import jax.numpy as jnp

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math_utils
import math_utils_jax 

def run_and_compare(
    test_name,
    jax_func,
    torch_func,
    args,
    atol=1e-7 # 使用更严格的容忍度
):
    print(f"--- Running Test: {test_name} ---")
    
    # prepare input
    torch_args = [torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
    jax_args = [jnp.array(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
    
    # convert output to np
    jax_output = np.array(jax_func(*jax_args))
    torch_output = torch_func(*torch_args).numpy()
    
    # compare shape and number
    assert jax_output.shape == torch_output.shape, f"{test_name} FAILED: Shape mismatch"
    assert np.allclose(jax_output, torch_output, atol=atol), f"{test_name} FAILED: Numeric mismatch"
    
    max_diff = np.max(np.abs(jax_output - torch_output))
    print(f"✅ PASSED: {test_name} (Max Diff: {max_diff:.2e})")
    return jax_output, torch_output

def test_modulate():
    B, N, C = 4, 256, 768
    x = np.random.randn(B, N, C).astype(np.float32)
    shift = np.random.randn(B, C).astype(np.float32)
    scale = np.random.randn(B, C).astype(np.float32)
    
    run_and_compare(
        "modulate",
        math_utils_jax.modulate,
        math_utils.modulate,
        (x, shift, scale)
    )

def test_1d_pos_embed():
    EMBED_DIM = 128
    LENGTH = 50 # 非完全平方数
    
    run_and_compare(
        "1D Positional Embedding",
        math_utils_jax.get_1d_sincos_pos_embed,
        math_utils.get_1d_sincos_pos_embed,
        (EMBED_DIM, LENGTH)
    )

def test_2d_pos_embed():   
    EMBED_DIM_1, GRID_SIZE_1 = 768, 16
    LENGTH_1 = GRID_SIZE_1 * GRID_SIZE_1
    jax_out_1, torch_out_1 = run_and_compare(
        "2D Positional Embedding (768, 16x16)",
        math_utils_jax.get_2d_sincos_pos_embed,
        math_utils.get_2d_sincos_pos_embed,
        (None, EMBED_DIM_1, LENGTH_1)
    )
    
    print("--- Checking specific positions for (768, 16x16) ---")
    quarter_dim_1 = EMBED_DIM_1 // 4

    assert np.all(torch_out_1[0, 0, :quarter_dim_1] == 0), "pos=0 sin part failed"
    assert np.allclose(torch_out_1[0, 0, quarter_dim_1:2*quarter_dim_1], 1), "pos=0 cos part failed"
    print("✅ Position 0 (start) checked.")

    mid_idx = LENGTH_1 // 2 + GRID_SIZE_1 // 2
    assert not np.allclose(torch_out_1[0, mid_idx, :5], 0), "Middle position should not be all zero"
    print(f"✅ Position {mid_idx} (middle) checked.")

    assert not np.allclose(torch_out_1[0, -1, :5], 0), "Last position should not be all zero"
    print(f"✅ Position {LENGTH_1-1} (end) checked.\n")

    EMBED_DIM_2, GRID_SIZE_2 = 512, 8
    LENGTH_2 = GRID_SIZE_2 * GRID_SIZE_2
    run_and_compare(
        "2D Positional Embedding (512, 8x8)",
        math_utils_jax.get_2d_sincos_pos_embed,
        math_utils.get_2d_sincos_pos_embed,
        (None, EMBED_DIM_2, LENGTH_2)
    )

if __name__ == "__main__":
    test_modulate()
    print("-" * 50)
    test_1d_pos_embed()
    print("-" * 50)
    test_2d_pos_embed()
    print("\n🎉 All tests passed successfully!")