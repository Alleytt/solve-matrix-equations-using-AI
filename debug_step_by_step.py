print("开始调试")

# 步骤1: 导入模块
print("\n步骤1: 导入模块")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# 步骤2: 定义类
print("\n步骤2: 定义类")
class LowRankContinuousMapping(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, latent_dim=20, output_shape=None, activation='sin'):
        super().__init__()
        self.output_shape = output_shape  # (n1, n2)
        self.latent_dim = latent_dim
        
        # MLP: p -> Φ(p)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if activation == 'sin':
            layers.append(self.Sin())
        else:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if activation == 'sin':
            layers.append(self.Sin())
        else:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)
        
        # 可学习 latent tensor C: n1 × n2 × latent_dim
        self.C = nn.Parameter(torch.randn(output_shape[0], output_shape[1], latent_dim))

    class Sin(nn.Module):
        def forward(self, x, omega=0.1):
            return torch.sin(omega * x)

    def forward(self, p):
        # p: (B, 1)
        phi = self.mlp(p)  # (B, latent_dim)
        # mode-3 tensor-matrix product: C ×3 phi
        B = phi.shape[0]
        n1, n2, d = self.C.shape
        C_reshaped = self.C.reshape(n1*n2, d)  # (n1n2, d)
        phi_reshaped = phi.unsqueeze(-1)  # (B, d, 1)
        out = torch.bmm(C_reshaped.unsqueeze(0).repeat(B,1,1), phi_reshaped)  # (B, n1n2, 1)
        out = out.reshape(B, n1, n2)
        return out

print("LowRankContinuousMapping 类定义完成")

class AlgebraicLoss(nn.Module):
    def __init__(self, op_type='inv', lambda_consist=1.0):
        super().__init__()
        self.op_type = op_type
        self.lambda_consist = lambda_consist
        self.mse = nn.MSELoss()

    def data_loss(self, pred, target):
        return self.mse(pred, target)

    def consistency_loss(self, A_p, pred):
        if self.op_type == 'inv':
            # 求逆约束: A(p) * A_inv(p) ≈ I
            I = torch.eye(A_p.shape[1], device=A_p.device).unsqueeze(0)
            product = torch.bmm(A_p, pred)
            return self.mse(product, I)
        elif self.op_type == 'svd':
            # SVD约束: A ≈ U S V^T, U^T U≈I, V^T V≈I
            U, S, Vt = pred
            recon = torch.bmm(torch.bmm(U, torch.diag_embed(S)), Vt)
            loss_recon = self.mse(recon, A_p)
            I = torch.eye(U.shape[1], device=U.device).unsqueeze(0)
            loss_ortho_u = self.mse(torch.bmm(U.transpose(1,2), U), I)
            loss_ortho_v = self.mse(torch.bmm(Vt.transpose(1,2), Vt), I)
            return loss_recon + loss_ortho_u + loss_ortho_v
        else:
            return 0.0

    def forward(self, pred, target, A_p=None):
        loss_data = self.data_loss(pred, target)
        loss_consist = self.consistency_loss(A_p, pred) if A_p is not None else 0.0
        return loss_data + self.lambda_consist * loss_consist

print("AlgebraicLoss 类定义完成")

# 步骤3: 定义函数
print("\n步骤3: 定义函数")
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
    return H

def get_ground_truth(p, n=256, op='inv'):
    H = generate_parametric_matrix(p, n)
    if op == 'inv':
        return torch.linalg.inv(H)
    elif op == 'svd':
        U, S, Vt = torch.linalg.svd(H)
        return U, S, Vt

print("函数定义完成")

# 步骤4: 测试矩阵生成
print("\n步骤4: 测试矩阵生成")
try:
    H = generate_parametric_matrix(0.5, n=8)
    print(f"生成矩阵形状: {H.shape}")
    print("矩阵生成成功")
except Exception as e:
    print(f"矩阵生成失败: {e}")

# 步骤5: 测试矩阵求逆
print("\n步骤5: 测试矩阵求逆")
try:
    gt = get_ground_truth(0.5, n=8, op='inv')
    print(f"逆矩阵形状: {gt.shape}")
    print("矩阵求逆成功")
except Exception as e:
    print(f"矩阵求逆失败: {e}")

# 步骤6: 测试模型创建
print("\n步骤6: 测试模型创建")
try:
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=100, latent_dim=20, 
                                     output_shape=(8, 8), activation='sin')
    print("模型创建成功")
except Exception as e:
    print(f"模型创建失败: {e}")

# 步骤7: 测试前向传播
print("\n步骤7: 测试前向传播")
try:
    p_tensor = torch.tensor([[0.5]], dtype=torch.float32)
    pred = model(p_tensor)
    print(f"模型输出形状: {pred.shape}")
    print("前向传播成功")
except Exception as e:
    print(f"前向传播失败: {e}")

print("\n调试完成")
