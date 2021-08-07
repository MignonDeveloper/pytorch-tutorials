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
print(my_tensor.dtype)  # torch.float32
print(my_tensor.type())  # torch.FloatTensor
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
# or you can use, tensor.type(torch.FloatTensor)
print(tensor.double()) # to torch.float64


# (4) Numpy Array to Tensor conversion and vice-versa
#       - Numpy를 Tensor로 바꿀 때, 혹은 Tensor를 Numpy로 바꿀 때
#       - 서로 포인터로 연결되어 있으므로 기존 객체의 변화가 반영됨
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)  # torch.from_numpy()
np_array_back = tensor.numpy()  # tensor.numpy()
np_array[:] = 1  # 기존 객체가 변하면 연결된 모든 객체도 변한다
print(tensor, np_array_back)


# pandas data to Tensor conversion
import pandas as pd
pandas_series = pd.Series([0.1, 2, 0.3, 10.1])
pandas_to_torch = torch.from_numpy(pandas_series.values)  # values를 통해 numpy array로 변환

# Tensor to list
tensor = torch.tensor([0, 1, 2, 3])
tensor_to_list = tensor.tolist()


# (5) Tensor deep copy
tensor = torch.tensor([0, 1, 2, 3])
new_tensor = tensor.clone()
tensor[0] = 100
print(tensor, new_tensor)