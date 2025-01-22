# 线性代数的实现——代码
import torch

# 标量
x = torch.tensor([3.0])
y = torch.tensor([2.0])

print(x + y)
print(x - y)
# 向量
x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)
# 矩阵
A = torch.arange(20).reshape(5, 4)

print(A)
print(A.T)

B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B.T)
print(B == B.T)

# 张量
X = torch.arange(24).reshape(2, 3, 4)
print(X)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A, A + B)

# 矩阵按元素乘法
print(A * B)

x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())

# 按维度求和
A = torch.arange(20 * 2).reshape(2, 5, 4)
print(A.shape, A.sum())
print(A)
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)
print(A.sum(axis=[0, 1]).shape)

# 平均值
A = torch.arange(20, dtype=torch.float32).reshape(2, 2, 5)
print(A.mean())
print(A.sum()/A.numel())
print(A)
print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])

# 保持维度不变
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

# 向量的点积
y = torch.ones(4, dtype=torch.float32)
x = torch.arange(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))

# 矩阵向量积
A = torch.arange(20,dtype=torch.float32).reshape(5, 4)
print(torch.mv(A,x))

# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

# 范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
# L1范数,向量元素的绝对值之和
print(torch.abs(u).sum())
# 矩阵的F范数,矩阵元素平方和的平方根
print(torch.norm(torch.ones((4, 9))))

# AXIS
a = torch.ones((2, 5, 4))
print(a.shape)
print(a.sum().shape)
print(a.sum(axis=[0,2], keepdims=True).shape)