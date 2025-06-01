import torch

t = torch.tensor([0., 1., 2., 3., 4., 5., 6.])
# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
