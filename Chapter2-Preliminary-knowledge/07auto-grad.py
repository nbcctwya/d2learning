import torch

x = torch.arange(4.0)
x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(x)
print(y)

# 调用 反向传播函数
y.backward()
print(x.grad)
print(x.grad == 4 * x)

# 梯度会累计，需要手动清零
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 对非标量调用'backward'需要传入一个‘gradient’参数
x.grad.zero_()
y = x * x
print(y)
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

# 移动到计算图外
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

# Python控制流或函数的求导
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)
print(d / a == a.grad)