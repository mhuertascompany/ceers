import torch
from pytorch_lightning.accelerators.cuda import CUDAAccelerator

print(CUDAAccelerator.is_available())
print(torch.__version__)

print(torch.cuda.is_available())