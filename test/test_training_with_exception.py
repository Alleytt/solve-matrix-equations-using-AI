print("开始测试训练过程")

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

    print("类定义完成")
    
    # 定义函数
    print("\n3. 定义函数")
    def generate_parametric_matrix(p, n=4, epsilon=1e-3):
        # 论文合成数据: H(p) = A(p)B(p)^T + εI
        p = p.item() if isinstance(p, torch.Tensor) else p
        A0 = torch.randn(n, 4)
        B0 = torch.randn(n, 4)
        FA = torch.rand(4) * 1.0 + 0.5
        FB = torch.rand(4) * 1.0 + 0.5
        PhiA = torch.rand(4) * 2 * np.pi
        PhiB = torch.rand(4) * 2 * np.pi
        
        A = A0 * torch.sin(2 * np.pi * FA * p + PhiA)
        B = B0 * torch.cos(2 * np.pi * FB * p + PhiB)
        H = A @ B.T + epsilon * torch.eye(n)
        return H

    def get_ground_truth(p, n=4, op='inv'):
        H = generate_parametric_matrix(p, n)
        if op == 'inv':
            return torch.linalg.inv(H)
        elif op == 'svd':
            U, S, Vt = torch.linalg.svd(H)
            return U, S, Vt

    print("函数定义完成")
    
    # 参数
    print("\n4. 设置参数")
    n = 4
    op = 'inv'
    num_train = 2
    max_iter = 3
    
    # 设备
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 采样
    print("\n5. 采样")
    p_train = np.random.uniform(0, 1, num_train)
    print(f"采样完成: p_train={p_train}")
    
    # 模型
    print("\n6. 创建模型")
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=10, latent_dim=5, 
                                     output_shape=(n, n), activation='sin').to(device)
    criterion = AlgebraicLoss(op_type=op, lambda_consist=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("模型创建完成")
    
    # 训练数据
    print("\n7. 准备训练数据")
    train_data = []
    for i, p in enumerate(p_train):
        print(f"处理第 {i+1}/{num_train} 个训练样本")
        H = generate_parametric_matrix(p, n).to(device)
        gt = get_ground_truth(p, n, op).to(device)
        # 确保gt的形状与模型输出匹配
        if op == 'inv':
            gt = gt.unsqueeze(0)  # 添加批次维度
        train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H, gt))
    print(f"训练数据准备完成: {len(train_data)} 个样本")
    
    # 训练
    print("\n8. 开始训练")
    for it in range(max_iter):
        print(f"\n第 {it} 轮迭代")
        model.train()
        total_loss = 0.0
        
        # 监督损失
        for i, (p_tensor, H_p, gt) in enumerate(train_data):
            print(f"处理第 {i+1}/{len(train_data)} 个样本")
            pred = model(p_tensor)
            loss = criterion(pred, gt, H_p.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"样本 {i+1} 损失: {loss.item():.4f}")
        
        print(f'迭代 {it}/{max_iter}, 总损失: {total_loss/num_train:.4f}')
    
    print("\n训练完成")
    print("测试成功完成")
    
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()
