import torch

# sets precision depending on availability of tensor cores
def set_smart_precision():
    if torch.cuda.is_available():
        # Check if the device is Ampere (8.0) or newer
        major, minor = torch.cuda.get_device_capability()
        print(f"Device capability {major}.{minor} detected.")
        if major >= 8:
            # Benefits from Tensor Cores; speedup justifies precision drop
            print(f"Using set_float32_matmul_precision('high').")
            torch.set_float32_matmul_precision('high')
        else:
            # Older GPU: Keep 'highest' to avoid precision loss without gain
            torch.set_float32_matmul_precision('highest')
    else:
        # CPU: Keep 'highest'
        torch.set_float32_matmul_precision('highest')
