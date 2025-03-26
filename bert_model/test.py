import torch

for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i)
        a = torch.randn(1000, device=f"cuda:{i}")
        print(f"GPU {i} is working.")
    except Exception as e:
        print(f"GPU {i} failed: {e}")
