import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
from typing import Dict, Any, Tuple
from einops import rearrange
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from diffusers import AutoencoderKL
from model import DiT
from utils.dataset import get_dataloader
import torch.optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# training config
DATASET_NAME = 'korexyz/celeba-hq-256x256'
DATASET_BASE_SAVE_PATH = "/home/jasonluo/code/shortcut-models-torch/data"
CKPT_BASE_SAVE_PATH = "/home/jasonluo/code/shortcut-models-torch/ckpt"
TOTAL_DENOISING_STEPS = 128 # 128
BATCH_SIZE = 32 # 16
BATCH_SPLIT_RATIO = 0.75 # in every batch, split FM : SC = 0.75 : 0.25
EMA_DECAY = 0.999
LEARNING_RATE = 1e-4 # 1e-4
TRAIN_STEPS = 100000
WEIGHT_DECAY = 0.1

# data preprocess config
PATCH_SIZE = 2
IMAGE_SIZE = 256

# model config
HIDDEN_SIZE = 768 # 768 must be num_head*n
DIT_DEPTH = 12 # 12
NUM_HEADS = 12
DROPOUT_RATE = 0.1
MLP_RATIO = 4
IN_CHANNELS = 4
OUT_CHANNELS = 4
NUM_CLASSES = 2

def update_ema(target_model, source_model, decay):
    """
    Update EMA model weights.
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)

def train_step(
    model: DiT,
    ema_model: DiT,
    optimizer: torch.optim.Optimizer,
    latents: torch.Tensor,
    labels: torch.Tensor,
    total_denoising_steps: int = TOTAL_DENOISING_STEPS,
    batch_split_ratio: float = BATCH_SPLIT_RATIO,
    ema_decay: float = EMA_DECAY
):
    m_inv = 1 / total_denoising_steps
    optimizer.zero_grad()
    B = latents.shape[0] # B = batch_size
    
    # sample
    z = torch.randn_like(latents).to(DEVICE)
    t = torch.rand(B, device=DEVICE) * (1.0 - m_inv) + m_inv # sampling t from [1/m_inv, 1]
    
    x1 = latents.to(DEVICE)
    x0 = z
    xt = (1 - t).view(B, 1, 1, 1) * x0 + t.view(B, 1, 1, 1) * x1
    
    v_target = x1 - x0
    
    fm_batch_num = int(B * batch_split_ratio)
    sc_batch_num = B - fm_batch_num
    
    # FM loss
    ## sampling
    fm_xt, fm_t, fm_y, fm_v_target = xt[:fm_batch_num], t[:fm_batch_num], labels[:fm_batch_num].to(DEVICE), v_target[:fm_batch_num]
    fm_dt = torch.zeros_like(fm_t)
    
    ## forward
    fm_s_theta = model(fm_xt, fm_t, fm_dt, fm_y)
    
    ## fm_loss
    fm_loss = (fm_s_theta - fm_v_target).pow(2).mean()
    
    total_loss = fm_loss
    loss_metrics = {'fm_loss': fm_loss.item()}
    
    # SC loss
    ## sampling
    sc_xt, sc_t, sc_y = xt[fm_batch_num:], t[fm_batch_num:], labels[fm_batch_num:].to(DEVICE)
    ## use binary recursive to sample dt, dt must be `2^k` times of `1/m_inv`
    T_max = int(math.log2(TOTAL_DENOISING_STEPS)) - 1
    log_d_idx = torch.randint(0, T_max, (sc_batch_num,), device=DEVICE).float()
    sc_dt = m_inv * (2.0 ** log_d_idx)
    sc_2dt = sc_dt * 2.0
    sc_t_times = torch.round(sc_t / sc_dt)
    sc_t = sc_t_times * sc_dt
    sc_xt = (1 - sc_t.view(sc_batch_num, 1, 1, 1)) * x0[fm_batch_num:] + sc_t.view(sc_batch_num, 1, 1, 1) * x1[fm_batch_num:]
    
    ## forward
    with torch.no_grad():
        sc_s1_ema = ema_model(sc_xt, sc_t, sc_dt, sc_y)
        sc_x1_ema = sc_xt + sc_s1_ema * sc_dt.view(sc_batch_num, 1, 1, 1)
        
        sc_s2_ema = ema_model(sc_x1_ema, sc_t + sc_dt, sc_dt, sc_y)
        sc_v_target_ema = (sc_s1_ema + sc_s2_ema) / 2
    
    sc_s_theta = model(sc_xt, sc_t, sc_2dt, sc_y)
    
    sc_loss = (sc_s_theta - sc_v_target_ema).pow(2).mean()
    total_loss += sc_loss
    loss_metrics['sc_loss'] = sc_loss.item()
    loss_metrics['total_loss'] = total_loss.item()
    
    # backward and update
    total_loss.backward()
    optimizer.step()
    
    update_ema(ema_model, model, ema_decay)
    
    return loss_metrics

def train():
    os.makedirs(CKPT_BASE_SAVE_PATH, exist_ok=True)
    
    model = DiT(
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout_rate=DROPOUT_RATE,
        dit_depth=DIT_DEPTH,
        out_channels=OUT_CHANNELS,
        ignore_dt=False
    ).to(DEVICE)
    
    ema_model = DiT(
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout_rate=DROPOUT_RATE,
        dit_depth=DIT_DEPTH,
        out_channels=OUT_CHANNELS,
        ignore_dt=False
    ).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    dataloader = get_dataloader(
        dataset_name=DATASET_NAME,
        batch_size=BATCH_SIZE,
        base_save_path=DATASET_BASE_SAVE_PATH,
        split="train",
        shuffle=True,
        num_workers=4,
        image_column="image",
        image_size=IMAGE_SIZE,
        preprocess_batch_size=32
    )
    
    model.train()
    pbar = tqdm(range(TRAIN_STEPS))
    step = 0
    
    for step in pbar:
        try:
            latents, labels = next(data_iter)
        except (NameError, StopIteration):
            data_iter = iter(dataloader)
            latents, labels = next(data_iter)
    
        loss_metrics = train_step(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            latents=latents,
            labels=labels,
            total_denoising_steps=TOTAL_DENOISING_STEPS,
            batch_split_ratio=BATCH_SPLIT_RATIO,
            ema_decay=EMA_DECAY
        )
        
        pbar.set_postfix({
            'total_loss': f"{loss_metrics['total_loss']:.4f}", 
            'fm_loss': f"{loss_metrics['fm_loss']:.4f}",
            'sc_loss': f"{loss_metrics.get('sc_loss', 0.0):.4f}"
        })
        
        if (step + 1) % 1000 == 0:
            print(f"\nStep {step + 1}: Saving checkpoint to '{CKPT_BASE_SAVE_PATH}'...")
            
            model_path = os.path.join(CKPT_BASE_SAVE_PATH, f"shortcut_model_step_{step+1}.pt")
            ema_model_path = os.path.join(CKPT_BASE_SAVE_PATH, f"shortcut_ema_model_step_{step+1}.pt")
            
            torch.save(model.state_dict(), model_path)
            torch.save(ema_model.state_dict(), ema_model_path)
        
    print("Training finished.")

if __name__ == '__main__':
    train()