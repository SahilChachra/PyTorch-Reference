import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t)

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(t)

# t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda")
# print(t)
# in output, shows device as 0 or cuda

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
print(t)
# in outputs, shows no device name, meaning its running on CPU

# device = "cuda" if torch.cuda.is_available() else "cpu"
# Now we can use the argument device = device when using tensors

# ++++++++ Attributes of Tensor ++++++++++
print('\n++++++++ Attributes of Tensor ++++++++++')

print('Dtype of Tensor :', t.dtype)
print('Device on which Tensor is residing :', t.device)
print('Shape of Tensor :', t.shape)
print('Req Grad : ', t.requires_grad)

# ++++++ TENSOR INITIALIZATION ++++++
print('\n++++++ TENSOR INITIALIZATION ++++++')

x = torch.empty(size=(3, 3))
print('torch.empty() : ', x)

x = torch.zeros(size=(4, 4))
print('torch.zeros() :', x)

x = torch.ones(size=(3, 4))
print('torch.ones() :', x)

x = torch.eye(5, 5)
print('torch.eye() :', x)

x = torch.arange(start=0, end=5, step=0.6)
print('torch.arange() :', x)

x = torch.linspace(start=0.1, end=1, steps=5)
print('torch.linspace() :', x)

x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)  # Normally distributed data
print('torch.empty().normal_(mean=0, std=1)', x)

x = torch.empty(size=(1, 5)).uniform_(0)
print('torch.empty().uniform_(0, 1) :', x)

x = torch.diag(torch.ones(3))
print('torch.diag(torch.ones(3)) :\n', x)

# ++++ Initialize Tensors to other types (int, float, double) +++++
print('\n++++ Initialize Tensors to other types (int, float, double) +++++')

t = torch.arange(4)
print('Tensor is :', t)
print('t.bool() :', t.bool())
print('t.short() :', t.short())
print('t.long() :', t.long())
print('t.half() :', t.half())  # Converts to float16
print('t.float() :', t.float())
print('t.double() :', t.double())

# ++++ Numpy & Tensor ++++
print('\n++++ Numpy & Tensor ++++')
import numpy as np

arr = np.zeros((5, 5))
print('Numpy Zeros 5x5 :', arr)
t = torch.from_numpy(arr)
print('Torch from numpy :', t)
np_arr = t.numpy()
print('Numpy from Torch :', np_arr)

# ++++ Tensor Math and Comparison Operations +++
print('\n++++ Tensor Math and Comparison Operations +++')

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

print('Addition : ', x + y)
print('Addition using torch.add() :', torch.add(x, y))

print('Subtraction :', x - y)

# Element wise division is same shape
z = torch.true_divide(x, y)  # x/y
print('Division : ', z)

# Inplace operations
t = torch.zeros(3)
t.add_(x)
print('t.add_(x) :', t)

# Exponential

z = x.pow(2)  # Element wise power of 2
print('x.pow(w) :', z)

z = x ** 2  # Same operation as above

# Simple operations
z = x > 0  # Element wise
print('x>0 :', z)

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)  # output will 2x3, or x3 = x1.mm(x2)
print('torch.mm(x1, x2) :\n', x3)

# Matrix Multiplication ( NOT ELEMENT WISE )

m = torch.rand(5, 5)
print('M :', m)
m.matrix_power(3)  # A x A x A
print('m.matrix_power(3) :', m)

# Elementwise

z = x * y
print('x * y : ', z)

# Dot product
z = torch.dot(x, y)
print('torch.dot() :', z)

# Batch Matrix Multiplication

batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand((batch, n, m))
t2 = torch.rand((batch, m, p))
out = torch.bmm(t1, t2)  # batch x n x p <- is output dimension
print('Batch Multiplication : ', out)

# Broadcasting

x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

del z

z = x1 - x2  # x2 row will be expanded so that it matches the rows of x1
# x2 will be subtracted with each row from x1

z = x1 ** x2  # x2 will become 5x5, each row repeated 5 times

# Other operations

sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)  # x.max(dim = 0)
values, indices = torch.min(x, dim=0)  # x.min(dim = 0)
abs_x = torch.abs(x)  # absolute value element wise
z = torch.argmax(x, dim = 0)
mean_x = torch.mean(x.float(), dim=0)

#Element wise compare
z = torch.eq(x, y)  # Returns True, False

sorted_z, indices = torch.sort(y, dim=0, descending=False)  # Sort the tensor

z = torch.clamp(x, min=0)  # Check all elements of X less than min(0 here) set it to 0
#  Like ReLU function

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
print(x)
x = torch.all(x)  # All of the values need to be True or >= 1

# +++++ Tensor Indexing +++++

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)  # x[0, :] -> 0th row, all columns(all features)
print(x[:, 0].shape)

print(x[2, 0:10])  # 0:10 -> [0, 1, 2, ..., 9]

# Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols]) # First selects rows then columns. Picks 1 row then 0th row

# Advance indexing

x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[(x > 2) & (x > 8)])
print(x[x.remainder(2)==0])

# Useful Operations

print(torch.where(x > 5, x, x*2))

print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())

print(x.ndimension())  # if 5x5x5 then it will return -> 5 Dim

print(x.numel()) # Count num of values

# ++++ Reshape Tensors ++++

x = torch.arange(9)

x_3x3 = x.view(3, 3)  # Both will work -> Acts on Contigous tensors
print('x_3x3 : ', x_3x3)
x_3x3 = x.reshape(3, 3)  # Same -> Acts, makes copy. Always work. Little slow

y = x_3x3.t()  #Transpose
print('Transpose of x_3x3 :', y)

    # - If view() doesnt work then the tensor is not contiguous. Then
print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)  # Concatenate
print(torch.cat((x1, x2), dim=1).shape)

    # Unroll the elements, have 10 elements instead of 2*5
z = x1.view(-1) # Flatten the tensor
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

  # Switch the axes. Keep batch but exchange 2 5 to 5 2
z = x.permute(0, 2, 1)  # keep 0 dimension to 0, 2nd dimension as 1st and 1st dimension as 2nd
print(z.shape)

x = torch.arange(10)
#  make 1x10 vector
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)  # to have 10x1 vector

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10

z = x.squeeze(0)  # to remove 1st 1
print(z.shape)