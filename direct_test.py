import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

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

def adaptive_sampling(model, A_func, candidate_p, epsilon_r=1e-3, epsilon_p=0.05, N_add=10):
    model.eval()
    with torch.no_grad():
        residuals = []
        for p in candidate_p:
            p_tensor = torch.tensor([[p]], dtype=torch.float32)
            pred = model(p_tensor)
            A_p = A_func(p).unsqueeze(0)
            # 计算残差
            I = torch.eye(A_p.shape[1]).unsqueeze(0)
            res = torch.norm(torch.bmm(A_p, pred) - I, p='fro').item()**2
            residuals.append(res)
    
    # 失败区域
    g = np.array(residuals) - epsilon_r
    fail_mask = g > 0
    fail_p = candidate_p[fail_mask]
    fail_res = g[fail_mask]
    
    # 失败概率
    fail_prob = len(fail_p) / len(candidate_p)
    
    if fail_prob > epsilon_p and len(fail_p) > 0:
        # 选残差最大的N_add个点
        idx = np.argsort(fail_res)[::-1][:N_add]
        new_col = fail_p[idx]
        return new_col, fail_prob
    else:
        return np.array([]), fail_prob

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

def train_neumatc(n=256, op='inv', num_train=40, num_test=100, max_iter=5000, update_T=500):
    print("进入 train_neumatc 函数")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 采样
    print("开始采样")
    p_train = np.random.uniform(0, 1, num_train)
    p_col = np.random.uniform(0, 1, 200)  # 初始配点
    p_candidate = np.random.uniform(0, 1, 1000)  # 候选采样
    print(f"采样完成: p_train={len(p_train)}, p_col={len(p_col)}, p_candidate={len(p_candidate)}")
    
    # 模型
    print("创建模型")
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=100, latent_dim=20, 
                                     output_shape=(n, n), activation='sin').to(device)
    criterion = AlgebraicLoss(op_type=op, lambda_consist=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("模型创建完成")
    
    # 训练数据
    print("准备训练数据")
    train_data = []
    for i, p in enumerate(p_train):
        print(f"处理第 {i+1}/{num_train} 个训练样本")
        H = generate_parametric_matrix(p, n).to(device)
        gt = get_ground_truth(p, n, op).to(device)
        train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H, gt))
    print(f"训练数据准备完成: {len(train_data)} 个样本")
    
    # 训练
    print(f"开始训练，共 {max_iter} 轮迭代")
    for it in range(max_iter):
        model.train()
        total_loss = 0.0
        
        # 监督损失
        for i, (p_tensor, H_p, gt) in enumerate(train_data):
            pred = model(p_tensor)
            loss = criterion(pred, gt, H_p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 自适应采样更新
        if it % update_T == 0 and it > 0:
            print(f"第 {it} 轮，进行自适应采样")
            new_col, fail_prob = adaptive_sampling(model, lambda p: generate_parametric_matrix(p,n), p_candidate)
            if len(new_col) > 0:
                p_col = np.concatenate([p_col, new_col])
                print(f"添加了 {len(new_col)} 个新样本")
        
        # 每10轮打印一次损失
        if it % 10 == 0:
            print(f'迭代 {it}/{max_iter}, 损失: {total_loss/num_train:.4f}')
    
    return model

def test_model(model, n=256, num_test=100, op='inv'):
    print("进入 test_model 函数")
    device = next(model.parameters()).device
    print(f"使用设备: {device}")
    p_test = np.random.uniform(0, 1, num_test)
    rel_err = 0.0
    
    # 推理时间
    start = time.time()
    
    with torch.no_grad():
        for i, p in enumerate(p_test):
            print(f"测试第 {i+1}/{num_test} 个样本")
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            H = generate_parametric_matrix(p, n).to(device)
            pred = model(p_tensor)
            gt = get_ground_truth(p, n, op).to(device)
            
            if op == 'inv':
                I = torch.eye(n).to(device)
                err = torch.norm(H @ pred - I) / torch.norm(I)
            elif op == 'svd':
                U, S, Vt = pred
                recon = torch.bmm(torch.bmm(U, torch.diag_embed(S)), Vt)
                err = torch.norm(recon - H) / torch.norm(H)
            rel_err += err.item()
    
    end = time.time()
    infer_time = (end - start) * 1000 / num_test  # ms per matrix
    rel_err /= num_test
    
    print(f'相对误差: {rel_err:.4e}')
    print(f'单矩阵推理时间: {infer_time:.2f} ms')
    return rel_err, infer_time

# 基线：NumPy直接求逆/SVD
def baseline_test(n=256, num_test=100, op='inv'):
    print("进入 baseline_test 函数")
    import time
    rel_err = 0.0
    start = time.time()
    for i in range(num_test):
        print(f"基线测试第 {i+1}/{num_test} 个样本")
        p = np.random.uniform(0,1)
        H = generate_parametric_matrix(p, n).numpy()
        if op == 'inv':
            gt = np.linalg.inv(H)
            pred = gt
            I = np.eye(n)
            err = np.linalg.norm(H @ pred - I) / np.linalg.norm(I)
        rel_err += err
    total_time = (time.time() - start) * 1000 / num_test
    rel_err /= num_test
    print(f'基线 相对误差: {rel_err:.4e}')
    print(f'基线 单矩阵时间: {total_time:.2f} ms')

# 测试代码
if __name__ == '__main__':
    # 超参数
    MATRIX_SIZE = 8
    OP_TYPE = 'inv'  # 'inv' 或 'svd'
    NUM_TRAIN = 2
    MAX_ITER = 10
    
    # 训练
    print("开始训练 NeuMatC...")
    model = train_neumatc(n=MATRIX_SIZE, op=OP_TYPE, num_train=NUM_TRAIN, max_iter=MAX_ITER)
    
    # 测试
    print("\nNeuMatC 测试:")
    test_model(model, n=MATRIX_SIZE, op=OP_TYPE, num_test=2)
    
    print("\nNumPy基线测试:")
    baseline_test(n=MATRIX_SIZE, op=OP_TYPE, num_test=2)
