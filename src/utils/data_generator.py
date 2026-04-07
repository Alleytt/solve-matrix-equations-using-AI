import torch
import numpy as np
from .matrix_error_handler import MatrixErrorHandler


def generate_parametric_matrix(p, n=256, epsilon=1e-3):
    # 论文合成数据: H(p) = A(p)B(p)^T + εI
    p = p.item() if isinstance(p, torch.Tensor) else p
    A0 = torch.randn(n, 64)
    B0 = torch.randn(n, 64)
    FA = torch.rand(64) * 1.0 + 0.5
    FB = torch.rand(64) * 1.0 + 0.5
    PhiA = torch.rand(64) * 2 * np.pi
    PhiB = torch.rand(64) * 2 * np.pi
    
    A = A0 * torch.sin(2 * np.pi * FA * p + PhiA)
    B = B0 * torch.cos(2 * np.pi * FB * p + PhiB)
    H = A @ B.T + epsilon * torch.eye(n)
    
    # 确保矩阵可逆
    while MatrixErrorHandler.is_singular(H):
        # 添加更多的正则化
        epsilon *= 10
        H = A @ B.T + epsilon * torch.eye(n)
        if epsilon > 1.0:
            # 如果仍然奇异，重新生成随机矩阵
            A0 = torch.randn(n, 64)
            B0 = torch.randn(n, 64)
            A = A0 * torch.sin(2 * np.pi * FA * p + PhiA)
            B = B0 * torch.cos(2 * np.pi * FB * p + PhiB)
            epsilon = 1e-3
            H = A @ B.T + epsilon * torch.eye(n)
    
    return H


def get_ground_truth(p, n=256, op='inv'):
    H = generate_parametric_matrix(p, n)
    if op == 'inv':
        try:
            # 使用更稳定的方法计算逆
            return torch.linalg.inv(H)
        except:
            # 处理奇异矩阵
            return MatrixErrorHandler.handle_singular_matrix(H)
    elif op == 'svd':
        try:
            U, S, Vt = torch.linalg.svd(H)
            return U, S, Vt
        except:
            raise ValueError("无法计算SVD")