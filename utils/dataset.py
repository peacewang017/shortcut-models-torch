# VAE based data pre-processing
# get dataloader

import torch
import os
from datasets import load_dataset
from diffusers import AutoencoderKL
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_from_disk

def preprocess_image_and_save(
    dataset_name: str,
    base_save_path: str,
    image_column: str = "image",
    preprocess_batch_size: int = 32,
    image_size: int = 256,
    split: str = "train"
) -> str:
    print("Loading VAE model...")
    vae_path = "stabilityai/sd-vae-ft-mse"
    vae = AutoencoderKL.from_pretrained(vae_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(device)
    vae.eval()
    vae_scale_factor = 0.18215
    
    preprocess = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    def transform_images_to_latents(batch):
        images = [preprocess(img.convert("RGB")) for img in batch[image_column]]
        image_batch = torch.stack(images).to(device)
        
        with torch.no_grad():
            latents_dist = vae.encode(image_batch).latent_dist
            latents = latents_dist.sample() * vae_scale_factor
        
        batch[image_column] = latents.cpu()
        return batch
    
    print(f"Loading and processing dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    
    processed_dataset = dataset.map(
        transform_images_to_latents,
        batched=True,
        batch_size=preprocess_batch_size
    )
    processed_dataset = processed_dataset.rename_column(image_column, "latents")
    
    save_path = os.path.join(base_save_path, dataset_name.replace("/", "_"))
    print(f"Saving updated dataset to {save_path}...")
    processed_dataset.save_to_disk(save_path)
    return save_path

def get_dataloader_after_preprocess(
    processed_dataset_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    print(f"Loading preprocessed latents from '{processed_dataset_path}'...")
    dataset = load_from_disk(processed_dataset_path)
    
    columns_to_set = ['latents']
    if 'label' in dataset.column_names:
        columns_to_set.append('label')
    dataset.set_format(type='torch', columns=columns_to_set) 
    
    def custom_collate(batch):
        latents = torch.stack([item['latents'] for item in batch])
        
        if 'label' in batch[0]:
            labels = torch.stack([item['label'] for item in batch])
        else:
            labels = torch.zeros(len(batch), dtype=torch.long)
            
        return latents, labels
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("DataLoader created successfully.")
    return dataloader
    
def get_dataloader(
    dataset_name: str,
    batch_size,
    base_save_path: str,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 4,
    image_column: str = "image",
    image_size: int = 256,
    preprocess_batch_size: int = 32,
):
    """
    Load image data from hugging face
    1. use a VAE preprocess
    2. "image" col -> "latent" col
    3. lazy load
    """
    os.makedirs(base_save_path, exist_ok=True)
    save_path = os.path.join(base_save_path, dataset_name.replace("/", "_"))
    
    # lazy load
    if os.path.exists(save_path):
        print(f"Found preprocessed dataset at '{save_path}', skipping preprocessing.")
    else:
        print(f"Preprocessed dataset not found at '{save_path}'. Starting preprocessing...")
        preprocess_image_and_save(
            dataset_name=dataset_name,
            base_save_path=base_save_path,
            image_column=image_column,
            preprocess_batch_size=preprocess_batch_size,
            image_size=image_size,
            split=split
        )
        
    return get_dataloader_after_preprocess(
        processed_dataset_path=save_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )