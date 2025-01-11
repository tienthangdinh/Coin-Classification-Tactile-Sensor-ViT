import torch

# Verify CUDA availability
print("Is CUDA available:", torch.cuda.is_available())

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Perform a tensor operation on the GPU
x = torch.randn(3, 3, device=device)
y = torch.randn(3, 3, device=device)
z = x + y
print("Result:", z)
