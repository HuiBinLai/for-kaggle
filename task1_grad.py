import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# 1. 定义卷积层和输入矩阵，完成一次前向传播
# 卷积层，为了求导方便，我们取 padding=0，并取消偏置项
conv_layer = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
# 输入矩阵
A = torch.randn(size=(1, 32, 32))
# 卷积后特征
Z = conv_layer(A)
# 为了实验方便，我们取 f(z) = sum(z)
sum_z = Z.sum()

# 2. Pytorch 自动微分机制计算卷积层的梯度
sum_z.backward()
print('Grad by Pytorch:', conv_layer.weight.grad)

# 3. 根据公式手动计算卷积层的梯度

# TODO
manual_grad = torch.zeros(size=(1, 1, 3, 3))
for i in range(3):
    for j in range(3):
        manual_grad[0, 0, i, j] = sum_z / Z[0, i, j]


print('Grad by manually computed:', manual_grad)
