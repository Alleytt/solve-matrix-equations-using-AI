print("开始测试")

# 步骤1: 导入模块
print("\n步骤1: 导入模块")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# 步骤2: 定义LowRankContinuousMapping类
print("\n步骤2: 定义LowRankContinuousMapping类")
class LowRankContinuousMapping(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10, latent_dim=5, output_shape=None, activation='sin'):
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

# 步骤3: 测试模型创建
print("\n步骤3: 测试模型创建")
try:
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=10, latent_dim=5, 
                                     output_shape=(4, 4), activation='sin')
    print("模型创建成功")
except Exception as e:
    print(f"模型创建失败: {e}")

# 步骤4: 测试前向传播
print("\n步骤4: 测试前向传播")
try:
    p_tensor = torch.tensor([[0.5]], dtype=torch.float32)
    pred = model(p_tensor)
    print(f"模型输出形状: {pred.shape}")
    print("前向传播成功")
except Exception as e:
    print(f"前向传播失败: {e}")

# 步骤5: 定义AlgebraicLoss类
print("\n步骤5: 定义AlgebraicLoss类")
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

# 步骤6: 测试损失计算
print("\n步骤6: 测试损失计算")
try:
    criterion = AlgebraicLoss(op_type='inv', lambda_consist=1.0)
    H = torch.randn(4, 4)
    H_inv = torch.linalg.inv(H)
    loss = criterion(pred, H_inv, H.unsqueeze(0))
    print(f"损失值: {loss.item()}")
    print("损失计算成功")
except Exception as e:
    print(f"损失计算失败: {e}")

# 步骤7: 测试优化器
print("\n步骤7: 测试优化器")
try:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("优化器创建成功")
except Exception as e:
    print(f"优化器创建失败: {e}")

# 步骤8: 测试反向传播
print("\n步骤8: 测试反向传播")
try:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("反向传播成功")
except Exception as e:
    print(f"反向传播失败: {e}")

print("\n测试完成")
