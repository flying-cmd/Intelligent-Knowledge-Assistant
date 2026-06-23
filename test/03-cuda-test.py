# test/03-cuda-test.py
try:
    import torch
    print(f"PyTorch loaded successfully. Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()} (False is normal for CPU-only builds)")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Failed to load PyTorch: {e}")
