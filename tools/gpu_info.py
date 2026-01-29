import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version (PyTorch): {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")
print(f"Supported Archs: {torch.cuda.get_arch_list()}")


# import torch
# print(torch.cuda.is_available())  # Should print: True
# print(torch.version.cuda)         # Should p rint: 11.8