import torch
import numpy as np

# 测试基本功能
print("测试基本功能")

# 测试矩阵生成
print("\n测试矩阵生成:")
n = 8
p = 0.5

# 生成矩阵
A0 = torch.randn(n, 64)
B0 = torch.randn(n, 64)
FA = torch.rand(64) * 1.0 + 0.5
FB = torch.rand(64) * 1.0 + 0.5
PhiA = torch.rand(64) * 2 * np.pi
PhiB = torch.rand(64) * 2 * np.pi

A = A0 * torch.sin(2 * np.pi * FA * p + PhiA)
B = B0 * torch.cos(2 * np.pi * FB * p + PhiB)
H = A @ B.T + 1e-3 * torch.eye(n)

print(f"生成的矩阵形状: {H.shape}")
print(f"矩阵前2x2:")
print(H[:2, :2])

# 测试矩阵求逆
print("\n测试矩阵求逆:")
H_inv = torch.linalg.inv(H)
print(f"逆矩阵形状: {H_inv.shape}")
print(f"逆矩阵前2x2:")
print(H_inv[:2, :2])

# 测试矩阵乘法
print("\n测试矩阵乘法:")
I = H @ H_inv
print(f"H @ H_inv 前2x2:")
print(I[:2, :2])

print("\n所有测试完成")
