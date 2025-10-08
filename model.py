# core shortcut DiT model

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from einops import rearrange
from utils.math import get_2d_sincos_pos_embed

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        # init model weight
        # same as JAX version `TrainConfig`
        self.apply(self._init_weights)
      
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, t:torch.Tensor) -> torch.Tensor:
        embedding = self.timestep_embedding(t)
        x = self.mlp(embedding)
        return x

    def timestep_embedding(self, t: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        dim = self.frequency_embedding_size
        half = dim // 2
        
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t.float()[:, None] * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.num_classes + 1,
            embedding_dim=self.hidden_size
        )
        
        # init model weight
        # same as JAX version `TrainConfig`
        with torch.no_grad():
            nn.init.normal_(self.embedding_layer.weight, std=0.02)
        
    def forward(self, labels: torch.Tensor):
        embeddings = self.embedding_layer(labels)
        return embeddings

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int, bias: bool = True):
        '''
        Args:
            patch_size: length of each patch
            in_chans: channel number of images
            embed_dim: dimension of each embedded vector, same as JAX version `PatchEmbed.hidden_size`
            bias: bias of conv_layer on / off 
        '''
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.conv_layer = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=bias
        )
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.conv_layer.weight)
            if self.conv_layer.bias is not None:
                nn.init.constant_(self.conv_layer.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        input:
            x: [B, C, H, W]
        output:
            [B, D, H_p, W_p] -> [B, N, D]
        """
        x = self.conv_layer(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class MlpBlock(nn.Module):
    """
    Transformer MLP / feed-forward block.
    """
    def __init__(self, in_dim: int, mlp_dim: int, out_dim: Optional[int] = None, dropout_rate: float = 0.1):
        """
        Args:
            in_dim: input dim
            mlp_dim: mlp hidden layer dim
            out_dim: output dim (if None, set same to in_dim)
            dropout_rate: Dropout rate, default 0.1
        """
        super().__init__()
        actual_out_dim = out_dim if out_dim is not None else in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, actual_out_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.apply(self._init_weights)
      
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)        
                        
    def forward(self, input):
        return self.mlp(input)
    
def modulate(x, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

################################################################################
#                                 Core DiT Model                               #
################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout_rate: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        
        self.adaLn_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.norm1 = nn.LayerNorm(
            normalized_shape=hidden_size, 
            eps=1e-6, 
            elementwise_affine=False
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=hidden_size, 
            eps=1e-6, 
            elementwise_affine=False
        )
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.final_proj = nn.Linear(hidden_size, hidden_size)
        
        self.mlp_layer = MlpBlock(
            in_dim=self.hidden_size,
            mlp_dim=int(self.hidden_size*self.mlp_ratio),
            out_dim=self.hidden_size,
            dropout_rate=self.dropout_rate      
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # adaLn forward
        c = self.adaLn_layer(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=-1)
        
        # multi-head attention
        x_norm = self.norm1(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        
        ## k, q, v shape: [B, num_heads, N, head_dim]
        ## attn_output: [B, N, hidden_size]
        k = rearrange(self.k_proj(x_modulated), 'b n (h d) -> b h n d', h=self.num_heads)
        q = rearrange(self.q_proj(x_modulated), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x_modulated), 'b n (h d) -> b h n d', h=self.num_heads)
        
        # attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k)
        attn_scores = attn_scores / self.head_dim
        attn_scores = attn_scores.float()
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output  = torch.einsum('bhij,bhjd->bhid', attn_weights, v).to(v.dtype)
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        attn_output = self.final_proj(attn_output)
        
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # MLP
        x_norm2 = self.norm2(x)
        x_modulated = modulate(x_norm2, shift_mlp, scale_mlp)
        
        mlp_x = self.mlp_layer(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_x
        
        return x
        
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, patch_size: int, out_channels: int, hidden_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size)
        )
        self.norm = nn.LayerNorm(
            normalized_shape=self.hidden_size,
            eps=1e-6, 
            elementwise_affine=False
        )
        self.proj = nn.Linear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  
        
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        c = self.mlp(c)
        shift, scale = c.chunk(2, dim=-1)
        
        x = self.norm(x)
        x = modulate(x, shift, scale)
        x = self.proj(x)
        return x
        
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    Accept and return Pytorch style image vectors [B, C, H, W].
    """
    def __init__(self, 
                 patch_size: int, 
                 hidden_size: int, 
                 in_channels: int,
                 num_classes: int,
                 num_heads: int,
                 mlp_ratio: int,
                 dropout_rate: int,
                 dit_depth: int,
                 out_channels: int,
                 ignore_dt: bool = False, 
                ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.ignore_dt = ignore_dt
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.dit_depth = dit_depth
        self.out_channels = out_channels
        
        self.patch_embed_layer = PatchEmbed(
            self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size,
        )
        self.time_embed_layer1 = TimestepEmbedder(
            hidden_size=self.hidden_size,
        )
        self.time_embed_layer2 = TimestepEmbedder(
            hidden_size=self.hidden_size,
        )
        self.label_embed_layer = LabelEmbedder(
            num_classes=self.num_classes,
            hidden_size=self.hidden_size
        )
        self.dit_blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio, self.dropout_rate)
            for _ in range(self.dit_depth)
        ])
        self.final_layer = FinalLayer(
            self.patch_size,
            out_channels=self.out_channels,
            hidden_size=self.hidden_size
        )
        
    def forward(self, x: torch.Tensor, t, dt, y):
        """
        Args:
            x = (Batch, Channels, Height, Width) image, x have to be squares
            t = (B,) timesteps, 
            y = (B,) class labels
        """
        
        batch_size = x.shape[0]
        input_size = x.shape[2]
        
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size

        if self.ignore_dt:
            dt = torch.zeros_like(t)
        
        # x (pos_embed & patch_embed)
        pos_embed = get_2d_sincos_pos_embed(None, self.hidden_size, num_patches).to(x.device) # pos_embed: [1, N=num_patches, D=hidden_size]
        x = self.patch_embed_layer(x)
        x = x + pos_embed # x: [B, N=num_patches, D=hidden_size]
        
        # c (timestep_embed & label_embed)
        ## convert {t, dt, y} into condition `c`
        t_embed = self.time_embed_layer1(t)
        dt_embed = self.time_embed_layer2(dt)
        y_embed = self.label_embed_layer(y)
        c = t_embed + dt_embed + y_embed # c: [B, D=hidden_size]
        
        # {x, c} -> x (dit blocks + final layer)
        for i, dit_block in enumerate(self.dit_blocks):
            x = dit_block(x, c)
        x = self.final_layer(x, c) # x [B, N, patch_size * patch_size * out_channels]
        
        # reshape x to images [Batch, Height, Width, out_channels]
        x = torch.reshape(x, 
                          (batch_size, num_patches_side, num_patches_side, self.patch_size, self.patch_size, self.out_channels)
                          )
        x = torch.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B C (H P) (W Q)', H=int(num_patches_side), W=int(num_patches_side))
        assert x.shape == (batch_size, self.out_channels, input_size, input_size)
        
        return x
        