import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()


print('data:', x.data)
print('grad_fn:', x.grad_fn)
print('grad:', x.grad)
print("is_leaf:", x.is_leaf)
print("requires_grad:", x.requires_grad)

print('data:', y.data)
print('grad_fn:', y.grad_fn)
print('grad:', y.grad)
print("is_leaf:", y.is_leaf)
print("requires_grad:", y.requires_grad)

###########################################

x = torch.tensor(2.0, requires_grad=True)
z = x ** 2 + 2 * x + 1
z.backward()
print(x.grad)


u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u * v + u ** 2
f.backward()
print(u.grad)
print(v.grad)