import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

def modulate(x, shift, scale):
    """
    Applies an affine transformation to the input tensor x.

    Args:
        x (torch.Tensor): Input tensor.
        shift (torch.Tensor): Shift vector.
        scale (torch.Tensor): Scale vector.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    # Clamp the scale to the range [-1, 1]
    scale = torch.clamp(scale, -1, 1)
    # Apply the modulation: x * (1 + scale) + shift
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generates 1D sinusoidal positional embeddings from a position grid.

    Args:
        embed_dim (int): The embedding dimension.
        pos (torch.Tensor): A tensor of positions.

    Returns:
        torch.Tensor: The positional embeddings.
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"
    
    # Calculate the omega frequencies
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    # Compute sine and cosine embeddings
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    # Concatenate sine and cosine embeddings
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Generates 1D sinusoidal positional embeddings for a given length.

    Args:
        embed_dim (int): The embedding dimension.
        length (int): The sequence length.

    Returns:
        torch.Tensor: Positional embeddings of shape (1, length, embed_dim).
    """
    # Create a position grid from 0 to length-1
    pos = torch.arange(length, dtype=torch.float32)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    # Add a batch dimension
    return emb.unsqueeze(0)

def get_2d_sincos_pos_embed(rng, embed_dim, length):
    """
    Generates 2D sinusoidal positional embeddings for a sequence,
    assuming it can be reshaped into a square grid.

    Args:
        rng: Unused parameter, kept for signature consistency.
        embed_dim (int): The embedding dimension.
        length (int): The sequence length, must be a perfect square.

    Returns:
        torch.Tensor: Positional embeddings of shape (1, length, embed_dim).
    """
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length, "Sequence length must be a perfect square"

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0, "Embedding dimension must be even"
        
        # Use half of the dimensions for height and half for width
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        
        # Concatenate height and width embeddings
        emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
        return emb

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    
    # Create a 2D grid; 'xy' indexing matches JAX/NumPy default behavior
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0) # Stack to shape (2, H, W)
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    # Add a batch dimension
    return pos_embed.unsqueeze(0) # (1, H*W, D)