import torch

# ============================================================== #
#                         Tensor Indexing                        #
# ============================================================== #


# (1) Basic Indexing
#      - slicing 역시 shallow copy이므로 새로운 tensor가 필요하면 clone 사용
batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)  # equivalent with x[0, :]
print(x[0, 0])  # it returns also tensor
print(x[0, 0].item())  # if want python integer then use item method
print(x[:, 0].shape)  # first feature of all the batch examples
print(x[2, 0:10].shape)
x[0, 0] = 100  # change the value by indexing


# (2) Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])  # indices list에 있는 index에 해당하는 요소를 반환

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])  # (1,4), (0,0)에 있는 요소를 반환


# (3) More advanced indexing
x = torch.arange(10)  # [0, 1, 2, ... ,9]

print(x[(x < 2) | (x > 8)])  # pick out all the elements less than 2 or greater than 8
print(x[(x > 2) & (x < 8)])  # pick out all the elements less than 2 and greater than 8
print(x[x.remainder(2) == 0])  # boolean indexing with conditions


# (4) Useful operations
print(torch.where(x > 5, x, x*2))  # if x > 5, change value to x, if not, change value to x * 2
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())  # get unique values
print(x.ndimension())  # rank of tensor, number of dimensionality
print(x.size())  # same with x.shape
print(x.numel())  # count number elements in tensor