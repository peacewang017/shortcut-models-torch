# VAE based data pre-processing
# dataloader

import torch
from datasets import load_dataset
from diffusers import AutoencoderKL
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(
    dataset_name: str,
    batch_size: int,
    image_size: int = 256
) -> DataLoader:
    """
    Args:
        dataset_name (str)
        batch_size
        image_size 
    """
    vae_path = "stabilityai/sd-vae-ft-mse"
    vae = AutoencoderKL.from_pretrained(vae_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(device)
    vae.eval()
    
    vae_scale_factor = 0.18215
    
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    if dataset_name.lower() == 'celeba-hq':
        hub_path = "korexyz/celeba-hq-256x256"
        image_column = "image"
        label_column = None
    elif dataset_name.lower() == 'imagenet':
        hub_path = "ILSVRC/imagenet-1k"
        image_column = "image"
        label_column = "label"
    else:
        raise ValueError(f"unsupported: {dataset_name}")
    
    print(f"loading dataset...")
    dataset = load_dataset(hub_path, split="train")
    
    def transform_function(examples):
        images = [preprocess(image.convert("RGB")) for image in examples[image_column]]
        image_batch = torch.stack(images)
        
        with torch.no_grad():
            latents_dist = vae.encode(image_batch.to(device)).latent_dist
            latents = latents_dist.sample() * vae_scale_factor
        
        output = {"latents": latents.cpu()}
        
        if label_column:
            output["labels"] = examples[label_column]
            
        return output
    
    def custom_collate(batch):
        # batch is a list of dictionaries, e.g., [{'latents': tensor, 'labels': int}]
        latents = torch.stack([item['latents'] for item in batch])
        
        if label_column:
            labels = torch.tensor([item['labels'] for item in batch])
            return latents, labels
        else:
            dummy_labels = torch.zeros(len(batch), dtype=torch.long)
            return latents, dummy_labels       
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=4
    )
    
    return dataloader
