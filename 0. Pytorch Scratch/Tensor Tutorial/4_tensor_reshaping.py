import torch

# ============================================================== #
#                        Tensor Reshaping                        #
# ============================================================== #


# (1) View & Reshape
x = torch.arange(9)
x_3x3 = x.view(3, 3)  # act on contiguous tensors = stored contiguously in memory
x_3x3 = x.reshape(3, 3)  # not matter -> if not contiguous, make copy of tensor

y = x_3x3.t()  # transpose matrix -> [0, 3, 6, 1, 4, 7, 2, 5, 8]
print(y.contiguous().view(9))  # get error without .contiguous()
print(y.reshape(9))  # safer way but may cost more computational operation


# (2) Concatenate
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)  # (4, 5)
print(torch.cat((x1, x2), dim=1).shape)  # (2, 10)


# (3) Unroll
z = x1.view(-1)  # flatten the entire things
print(z.shape)  # 10

batch = 64
x = torch.rand((batch, 2, 5, 5))
z = x.view(batch, -1)
print(z.shape)  # (64, 50)


# (4) permutation
z = x.permute(0, 2, 3, 1)
print(z.shape)  # (64, 5, 5, 2)


# (5) Squeeze, Unsqueeze
x = torch.arange(10)
print(x.unsqueeze(0).shape)  # (1, 10)

x = torch.rand((1, 1, 10))  # (1, 1, 10)
print(x.squeeze(1).shape)  # (1, 10) 