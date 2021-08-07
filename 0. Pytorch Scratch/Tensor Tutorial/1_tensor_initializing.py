import torch

# ============================================== #
#               Initializing Tensor              #
# ============================================== #

# (1) 초기값으로 Tensor 만들기 (with various parameters)
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor(data=[[1, 2, 3], [4, 5, 6]],   # Tensor Data in List
                         dtype=torch.float32,           # data type
                         device=device,                 # memory type
                         requires_grad=True)            # require grads for autograd (computational tree)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)  # if multiple GPU -> show where is tensor
print(my_tensor.shape)
print(my_tensor.requires_grad)


# (2) Other common initialization methods
empty_x = torch.empty(size=(3, 3))  # if we don't have exact value at time -> random values
zeros_x = torch.zeros((3, 3))
rand_x = torch.rand((3, 3))
ones_x = torch.ones((3, 3))
eye_x = torch.eye(5, 5)  # I: identity matrix
range_x = torch.arange(start=0, end=5, step=1)
linspace_x = torch.linspace(start=0.1, end=1, steps=10)
normal_x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)  # normal distribution
uniform_x = torch.empty(size=(1, 5)).uniform_(0, 1)  # uniform distribution b/w (from, to)
diag_x = torch.diag(torch.ones(3))  # 기존 Tensor의 대각선에 갖고 있던 value를 모두 유지하면서 diagonal matrix를 만든다.

print(empty_x)
print(zeros_x)
print(rand_x)
print(ones_x)
print(eye_x)
print(range_x)
print(linspace_x)
print(normal_x)
print(uniform_x)
print(diag_x)


# (3) How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)  # [0, 1, 2, 3]
print(tensor.bool())  # [False, True, True, True]
print(tensor.short())  # to torch.int16
print(tensor.long())  # to torch.int64
print(tensor.half())  # to torch.float16
print(tensor.float())  # to torch.float32
print(tensor.double()) # to torch.float64


# (4) Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)  # torch.from_numpy
np_array_back = tensor.numpy()  # tensor.numpy()