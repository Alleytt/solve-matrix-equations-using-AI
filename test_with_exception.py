print("开始测试")

try:
    # 导入模块
    print("\n1. 导入模块")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import time
    print("模块导入成功")
    
    # 定义类
    print("\n2. 定义类")
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
    
    # 创建模型
    print("\n3. 创建模型")
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=10, latent_dim=5, 
                                     output_shape=(4, 4), activation='sin')
    print("模型创建成功")
    
    # 测试前向传播
    print("\n4. 测试前向传播")
    p_tensor = torch.tensor([[0.5]], dtype=torch.float32)
    pred = model(p_tensor)
    print(f"模型输出形状: {pred.shape}")
    print("前向传播成功")
    
    print("\n测试成功完成")
    
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()
