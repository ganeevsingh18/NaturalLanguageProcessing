# import torch

# # Check if CUDA is available
# print(torch.cuda.is_available())

# # If you want to know how many GPUs
# print(torch.cuda.device_count())

# # To get the name of the current GPU
# if torch.cuda.is_available():
#     print(torch.cuda.get_device_name(0))
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create two random tensors
a = torch.rand(10000, 10000, device=device)
b = torch.rand(10000, 10000, device=device)

# Perform matrix multiplication on the GPU
c = torch.matmul(a, b)

# Move result back to CPU (if needed)
c_cpu = c.to("cpu")

print("Computation finished, result shape:", c_cpu.shape)
