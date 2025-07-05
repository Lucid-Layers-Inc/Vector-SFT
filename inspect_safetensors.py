import torch
from safetensors import safe_open

with safe_open("VectorSFT-checkpoints-test/checkpoint-10/adapter_model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        size_bytes = tensor.numel() * tensor.element_size()
        size_mb = size_bytes / (1024**2)
        
        print(f"Key: {key}, Size: {size_mb:.2f} MB")