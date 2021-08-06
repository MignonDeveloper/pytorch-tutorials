import torch

# ============================================================== #
#               Tensor Math & Comparison Operations              #
# ============================================================== #

# (0) Initialize Tensor
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])


# (1) Addition (with various ways)
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)

z = x + y  # favorite way


# (2) Subtraction
z = x - y


# (3) Division
z = torch.true_divide(x, y)  # element-wise division if got equal shape of vectors


# (4) inplace operations -> any operation ends with _ mean inplace operation, more efficient way
t = torch.zeros(3)
t.add_(x)
t += x  # 이것과는 달리 t = t + x는 먼저 copy가 일어난다. 즉, inplace operation이 아님


# (5) Exponentiation
z = x.pow(2)  # element-wise power operation
z = x ** 2


# (6) Simple Comparison
z = x > 0
z = x < 0


# (7) Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2 x 3
x3 = x1.mm(x2)


# (8) Matrix Exponentiation (not element-wise exponentiation)
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)  # 5x5 matrix를 3번 행렬곱 연산을 진행


# (9) element wise multiplication
z = x * y


# (10) dot product
z = torch.dot(x, y)


# (11) Batch Matrix Multiplication (batch 단위로 마지막 2개의 matrix에 대한 행렬곱 연산)
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)


# (12) Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2
z = x1 ** x2


# (13) Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)  # torch needs float tensors for mean
z = torch.eq(x, y)  # which elements is equal and return boolean values
sorted_y, indices = torch.sort(y, dim=0, descending=False)
z = torch.clamp(x, min=0, max=10)  # check all elements with speicified value and clamp it to the min & max value

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)  # return true if any value is true
z = torch.all(x)  # return true if all values are true