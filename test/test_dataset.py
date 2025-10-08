import sys
import os
import torch
import tempfile
from datasets import load_from_disk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import preprocess_image_and_save, get_dataloader


def test_preprocess_image_and_save(dataset_name, base_save_path, image_size) -> str:
    save_path = os.path.join(base_save_path, dataset_name.replace("/", "_"))
    
    test_split = "train[:4]"
    test_batch_size = 2

    save_path = preprocess_image_and_save(
        dataset_name=dataset_name,
        base_save_path=base_save_path,
        image_size=image_size,
        split=test_split,
        batch_size=test_batch_size
    )

    loaded_dataset = load_from_disk(save_path)
    loaded_dataset.set_format(type='torch', columns=['latents'])
    
    print("\n--- Verification Results ---")
    
    # Requirement 2: Print all columns in the on-disk dataset
    print(f"Columns in the saved dataset: {loaded_dataset.column_names}")

    sample = loaded_dataset[0]
    latents_tensor = sample["latents"]
    
    # Requirement 1: Verify the latent shape based on image_size
    downsample_factor = 8
    expected_latent_dim = image_size // downsample_factor
    expected_shape = (4, expected_latent_dim, expected_latent_dim)
    
    assert latents_tensor.shape == expected_shape, \
        f"Latent shape mismatch. Expected {expected_shape}, but got {latents_tensor.shape}."
    
    print(f"Latent shape verification passed. Shape is {latents_tensor.shape} as expected.")
    print("✅ Test validation successful.")
    return save_path

def test_dataloader():
    dataset_name = "korexyz/celeba-hq-256x256"
    base_save_path = "/home/jasonluo/code/shortcut-models-torch/data" # <-- 硬编码的路径
    image_size = 256
    num_samples_to_process = 100
    preprocess_batch_size = 16
    loader_batch_size = 100

    my_dataloader = get_dataloader(
        dataset_name=dataset_name,
        base_save_path=base_save_path,
        batch_size=loader_batch_size,
        preprocess_batch_size=preprocess_batch_size,
        image_size=image_size,
        split=f"train[:{num_samples_to_process}]",
    )
    
    print("\n--- Verifying DataLoader Output ---")
    
    latents_batch, labels_batch = next(iter(my_dataloader))
    
    print(f"Successfully fetched one batch of data.")
    print(f"Requested batch size: {loader_batch_size}")
    print(f"Latents batch shape: {latents_batch.shape}")
    print(f"Latents batch dtype: {latents_batch.dtype}")
    print(f"Labels batch shape: {labels_batch.shape}")
    print(f"Labels batch dtype: {labels_batch.dtype}")

    assert latents_batch.shape[0] == loader_batch_size
    assert latents_batch.shape[1:] == (4, image_size // 8, image_size // 8)
    
    print("\n✅ Full pipeline test successful!")

if __name__ == '__main__':
    try:
        # save_path = test_preprocess_image_and_save(
        #     dataset_name="korexyz/celeba-hq-256x256",
        #     base_save_path="/home/jasonluo/code/shortcut-models-torch/data",
        #     image_size=256
        # )
        # print(save_path)
        
        
        # test_preprocess_image_and_save(
        #     dataset_name="ILSVRC/imagenet-1k",
        #     base_save_path=temp_dir,
        #     image_size=256
        # )
        
        test_dataloader()
    except Exception as e:
        print(f"\n❌ An error occurred during validation: {e}")