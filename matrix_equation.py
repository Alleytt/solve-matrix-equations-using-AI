import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MatrixErrorHandler:
    """矩阵错误处理和鲁棒性模块"""
    
    @staticmethod
    def is_singular(matrix, tol=1e-10):
        """检测矩阵是否奇异"""
        if matrix.shape[0] != matrix.shape[1]:
            return True  # 非方阵
        
        try:
            det = torch.linalg.det(matrix)
            return abs(det) < tol
        except:
            return True
    
    @staticmethod
    def get_rank(matrix, tol=1e-10):
        """计算矩阵的秩"""
        try:
            _, S, _ = torch.linalg.svd(matrix)
            rank = torch.sum(S > tol).item()
            return rank
        except:
            return 0
    
    @staticmethod
    def analyze_equation(A, B, equation_type='AX=B'):
        """分析方程解的类型"""
        if equation_type == 'AX=B':
            # AX=B
            rank_A = MatrixErrorHandler.get_rank(A)
            rank_Aug = MatrixErrorHandler.get_rank(torch.cat([A, B], dim=1))
            
            if rank_A < rank_Aug:
                return "无解"
            elif rank_A == rank_Aug == A.shape[1]:
                return "唯一解"
            else:
                return "无穷多解"
        elif equation_type == 'XA=B':
            # XA=B -> A^T X^T = B^T
            return MatrixErrorHandler.analyze_equation(A.T, B.T, 'AX=B')
        elif equation_type == 'AXB=C':
            # AXB=C -> 先求解AY=C，再求解 Y=XB
            if not MatrixErrorHandler.is_singular(A) and not MatrixErrorHandler.is_singular(B):
                return "唯一解"
            else:
                return "可能有无解或无穷多解"
        else:
            return "未知"
    
    @staticmethod
    def handle_singular_matrix(matrix):
        """处理奇异矩阵，返回伪逆"""
        try:
            return torch.linalg.pinv(matrix)
        except:
            raise ValueError("无法处理奇异矩阵")

class MatrixAnalyzer:
    """矩阵分析和智能算法选择模块"""
    
    @staticmethod
    def is_positive_definite(matrix, tol=1e-8):
        """检测矩阵是否正定"""
        try:
            # 尝试Cholesky分解
            torch.linalg.cholesky(matrix)
            return True
        except:
            return False
    
    @staticmethod
    def condition_number(matrix):
        """计算矩阵的条件数"""
        try:
            _, S, _ = torch.linalg.svd(matrix)
            if len(S) < 2:
                return 1.0
            return S.max().item() / S.min().item()
        except:
            return float('inf')
    
    @staticmethod
    def is_ill_conditioned(matrix, threshold=1e6):
        """检测矩阵是否病态"""
        cond = MatrixAnalyzer.condition_number(matrix)
        return cond > threshold
    
    @staticmethod
    def select_solver(matrix, equation_type='AX=B'):
        """根据矩阵特性选择最佳求解方法"""
        n = matrix.shape[0]
        
        # 检查矩阵是否奇异
        if MatrixErrorHandler.is_singular(matrix):
            return "pinv"  # 使用伪逆
        
        # 检查矩阵是否正定
        if MatrixAnalyzer.is_positive_definite(matrix):
            return "cholesky"  # 使用Cholesky分解
        
        # 检查矩阵是否病态
        if MatrixAnalyzer.is_ill_conditioned(matrix):
            return "svd"  # 使用SVD
        
        # 根据矩阵大小选择方法
        if n < 100:
            return "inv"  # 直接求逆
        elif n < 1000:
            return "lu"  # 使用LU分解
        else:
            return "qr"  # 使用QR分解
    
    @staticmethod
    def solve_with_selector(A, B, equation_type='AX=B'):
        """使用智能选择的方法求解线性方程组"""
        solver = MatrixAnalyzer.select_solver(A, equation_type)
        print(f"使用求解方法: {solver}")
        
        if solver == "inv":
            return torch.linalg.inv(A) @ B
        elif solver == "cholesky":
            try:
                L = torch.linalg.cholesky(A)
                return torch.cholesky_solve(B, L)
            except:
                return torch.linalg.solve(A, B)
        elif solver == "lu":
            try:
                LU, pivots = torch.linalg.lu_factor(A)
                return torch.linalg.lu_solve(LU, pivots, B)
            except:
                return torch.linalg.solve(A, B)
        elif solver == "qr":
            try:
                Q, R = torch.linalg.qr(A)
                return torch.linalg.solve(R, Q.T @ B)
            except:
                return torch.linalg.solve(A, B)
        elif solver == "svd":
            U, S, Vt = torch.linalg.svd(A)
            S_inv = torch.diag_embed(1.0 / S)
            return Vt.T @ S_inv @ U.T @ B
        elif solver == "pinv":
            return torch.linalg.pinv(A) @ B
        else:
            return torch.linalg.solve(A, B)

class MatrixExplainer:
    """矩阵方程求解解释模块"""
    
    @staticmethod
    def explain_equation(equation_type, A, B=None, C=None):
        """解释方程类型和求解思路"""
        explanation = f"求解方程类型: {equation_type}\n"
        
        if equation_type == 'AX=B':
            explanation += "求解思路: 找到矩阵X，使得A与X的乘积等于B\n"
            explanation += f"矩阵A形状: {A.shape}, 矩阵B形状: {B.shape}\n"
            explanation += "求解方法: X = A^{-1}B (当A可逆时)"
        elif equation_type == 'XA=B':
            explanation += "求解思路: 找到矩阵X，使得X与A的乘积等于B\n"
            explanation += f"矩阵A形状: {A.shape}, 矩阵B形状: {B.shape}\n"
            explanation += "求解方法: X = BA^{-1} (当A可逆时)"
        elif equation_type == 'AXB=C':
            explanation += "求解思路: 找到矩阵X，使得A、X、B的乘积等于C\n"
            explanation += f"矩阵A形状: {A.shape}, 矩阵B形状: {B.shape}, 矩阵C形状: {C.shape}\n"
            explanation += "求解方法: X = A^{-1}CB^{-1} (当A和B都可逆时)"
        elif equation_type == 'inv':
            explanation += "求解思路: 找到矩阵A的逆矩阵A^{-1}\n"
            explanation += f"矩阵A形状: {A.shape}\n"
            explanation += "求解方法: A^{-1} (当A可逆时)"
        
        return explanation
    
    @staticmethod
    def explain_solution_steps(equation_type, A, B=None, C=None, X=None):
        """解释求解步骤"""
        steps = []
        
        if equation_type == 'AX=B':
            steps.append("步骤1: 检查矩阵A是否可逆")
            steps.append("步骤2: 计算矩阵A的逆矩阵A^{-1}")
            steps.append("步骤3: 计算X = A^{-1}B")
            steps.append("步骤4: 验证AX = B")
        elif equation_type == 'XA=B':
            steps.append("步骤1: 检查矩阵A是否可逆")
            steps.append("步骤2: 计算矩阵A的逆矩阵A^{-1}")
            steps.append("步骤3: 计算X = BA^{-1}")
            steps.append("步骤4: 验证XA = B")
        elif equation_type == 'AXB=C':
            steps.append("步骤1: 检查矩阵A和B是否可逆")
            steps.append("步骤2: 计算矩阵A的逆矩阵A^{-1}")
            steps.append("步骤3: 计算矩阵B的逆矩阵B^{-1}")
            steps.append("步骤4: 计算X = A^{-1}CB^{-1}")
            steps.append("步骤5: 验证AXB = C")
        elif equation_type == 'inv':
            steps.append("步骤1: 检查矩阵A是否可逆")
            steps.append("步骤2: 计算矩阵A的逆矩阵A^{-1}")
            steps.append("步骤3: 验证AA^{-1} = I")
        
        return steps
    
    @staticmethod
    def explain_matrix_properties(A):
        """解释矩阵的性质"""
        properties = []
        
        # 检查是否为方阵
        if A.shape[0] == A.shape[1]:
            properties.append(f"矩阵是方阵，大小为 {A.shape[0]}×{A.shape[1]}")
        else:
            properties.append(f"矩阵是长方形，大小为 {A.shape[0]}×{A.shape[1]}")
        
        # 检查是否奇异
        if MatrixErrorHandler.is_singular(A):
            properties.append("矩阵是奇异的，不可逆")
        else:
            properties.append("矩阵是非奇异的，可逆")
        
        # 检查是否正定
        if MatrixAnalyzer.is_positive_definite(A):
            properties.append("矩阵是正定的")
        
        # 计算条件数
        cond = MatrixAnalyzer.condition_number(A)
        properties.append(f"矩阵的条件数: {cond:.2f}")
        if MatrixAnalyzer.is_ill_conditioned(A):
            properties.append("矩阵是病态的")
        
        # 计算秩
        rank = MatrixErrorHandler.get_rank(A)
        properties.append(f"矩阵的秩: {rank}")
        
        return properties
    
    @staticmethod
    def generate_explanation(equation_type, A, B=None, C=None, X=None):
        """生成完整的解释"""
        explanation = "# 矩阵方程求解解释\n\n"
        
        # 方程类型和求解思路
        explanation += "## 1. 方程信息\n"
        explanation += MatrixExplainer.explain_equation(equation_type, A, B, C)
        explanation += "\n\n"
        
        # 矩阵性质
        explanation += "## 2. 矩阵性质\n"
        properties = MatrixExplainer.explain_matrix_properties(A)
        for prop in properties:
            explanation += f"- {prop}\n"
        explanation += "\n"
        
        # 求解步骤
        explanation += "## 3. 求解步骤\n"
        steps = MatrixExplainer.explain_solution_steps(equation_type, A, B, C, X)
        for i, step in enumerate(steps, 1):
            explanation += f"{i}. {step}\n"
        explanation += "\n"
        
        # 解的验证
        if X is not None:
            explanation += "## 4. 解的验证\n"
            if equation_type == 'AX=B':
                AX = A @ X
                error = torch.norm(AX - B).item()
                explanation += f"计算AX: 形状 = {AX.shape}\n"
                explanation += f"验证误差: ||AX - B|| = {error:.4e}\n"
            elif equation_type == 'XA=B':
                XA = X @ A
                error = torch.norm(XA - B).item()
                explanation += f"计算XA: 形状 = {XA.shape}\n"
                explanation += f"验证误差: ||XA - B|| = {error:.4e}\n"
            elif equation_type == 'AXB=C':
                AXB = A @ X @ B
                error = torch.norm(AXB - C).item()
                explanation += f"计算AXB: 形状 = {AXB.shape}\n"
                explanation += f"验证误差: ||AXB - C|| = {error:.4e}\n"
            elif equation_type == 'inv':
                AA_inv = A @ X
                I = torch.eye(A.shape[0])
                error = torch.norm(AA_inv - I).item()
                explanation += f"计算AA^{-1}: 形状 = {AA_inv.shape}\n"
                explanation += f"验证误差: ||AA^{-1} - I|| = {error:.4e}\n"
        
        return explanation
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
    def __init__(self, op_type='inv', lambda_consist=1.0, equation_type='AX=B'):
        super().__init__()
        self.op_type = op_type
        self.equation_type = equation_type
        self.lambda_consist = lambda_consist
        self.mse = nn.MSELoss()

    def data_loss(self, pred, target):
        return self.mse(pred, target)

    def consistency_loss(self, A_p, pred, B_p=None, C_p=None):
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
            I = torch.eye(U.shape[1], device=A_p.device).unsqueeze(0)
            loss_ortho_u = self.mse(torch.bmm(U.transpose(1,2), U), I)
            loss_ortho_v = self.mse(torch.bmm(Vt.transpose(1,2), Vt), I)
            return loss_recon + loss_ortho_u + loss_ortho_v
        elif self.equation_type == 'AX=B' and B_p is not None:
            # AX=B约束: A(p) * X(p) ≈ B(p)
            product = torch.bmm(A_p, pred)
            return self.mse(product, B_p)
        elif self.equation_type == 'XA=B' and B_p is not None:
            # XA=B约束: X(p) * A(p) ≈ B(p)
            product = torch.bmm(pred, A_p)
            return self.mse(product, B_p)
        elif self.equation_type == 'AXB=C' and B_p is not None and C_p is not None:
            # AXB=C约束: A(p) * X(p) * B(p) ≈ C(p)
            product = torch.bmm(torch.bmm(A_p, pred), B_p)
            return self.mse(product, C_p)
        else:
            return 0.0

    def forward(self, pred, target, A_p=None, B_p=None, C_p=None):
        loss_data = self.data_loss(pred, target)
        loss_consist = self.consistency_loss(A_p, pred, B_p, C_p) if A_p is not None else 0.0
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

def solve_linear_system(A, B, equation_type='AX=B'):
    """使用智能选择的方法求解线性方程组 AX=B"""
    try:
        # 使用智能算法调度
        return MatrixAnalyzer.solve_with_selector(A, B, equation_type)
    except:
        # 处理奇异矩阵
        A_pinv = MatrixErrorHandler.handle_singular_matrix(A)
        return A_pinv @ B
def train_neumatc(n=256, op='inv', equation_type='AX=B', num_train=40, num_test=100, max_iter=5000, update_T=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 采样
    p_train = np.random.uniform(0, 1, num_train)
    p_col = np.random.uniform(0, 1, 200)  # 初始配点
    p_candidate = np.random.uniform(0, 1, 1000)  # 候选采样
    
    # 模型
    model = LowRankContinuousMapping(input_dim=1, hidden_dim=100, latent_dim=20, 
                                     output_shape=(n, n), activation='sin').to(device)
    criterion = AlgebraicLoss(op_type=op, lambda_consist=1.0, equation_type=equation_type)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练数据
    train_data = []
    for p in p_train:
        H = generate_parametric_matrix(p, n).to(device)
        
        # 检查矩阵奇异性
        if MatrixErrorHandler.is_singular(H):
            print(f"警告: 生成的矩阵在参数p={p}时是奇异的，已使用正则化处理")
        
        # 根据方程类型生成不同的训练数据
        if equation_type == 'AX=B':
            # 生成随机B，然后使用智能方法求解X=A^{-1}B
            B = torch.randn(n, n, device=device)
            X = solve_linear_system(H, B, equation_type)
            # 分析方程解的类型
            solution_type = MatrixErrorHandler.analyze_equation(H, B, equation_type)
            if solution_type != "唯一解":
                print(f"警告: 方程在参数p={p}时{solution_type}")
            gt = X.unsqueeze(0).float()  # X是解
            train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), B.unsqueeze(0), None, gt))
        elif equation_type == 'XA=B':
            # 生成随机B，然后使用智能方法求解X=BA^{-1}
            B = torch.randn(n, n, device=device)
            X = solve_linear_system(H.T, B.T, 'AX=B').T  # XA=B -> A^T X^T = B^T
            # 分析方程解的类型
            solution_type = MatrixErrorHandler.analyze_equation(H, B, equation_type)
            if solution_type != "唯一解":
                print(f"警告: 方程在参数p={p}时{solution_type}")
            gt = X.unsqueeze(0).float()  # X是解
            train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), B.unsqueeze(0), None, gt))
        elif equation_type == 'AXB=C':
            # 生成随机C和B矩阵，然后使用智能方法求解X=A^{-1}CB^{-1}
            B_mat = generate_parametric_matrix(p + 0.1, n).to(device)  # 不同参数的矩阵
            C = torch.randn(n, n, device=device)
            # 先求解AY=C
            Y = solve_linear_system(H, C, 'AX=B')
            # 再求解XB=Y -> X=YB^{-1}
            X = solve_linear_system(B_mat.T, Y.T, 'AX=B').T
            # 分析方程解的类型
            solution_type = MatrixErrorHandler.analyze_equation(H, B_mat, equation_type)
            if solution_type != "唯一解":
                print(f"警告: 方程在参数p={p}时{solution_type}")
            gt = X.unsqueeze(0).float()  # X是解
            train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), B_mat.unsqueeze(0), C.unsqueeze(0), gt))
        else:  # 传统的求逆
            gt = get_ground_truth(p, n, op).to(device)
            if op == 'inv':
                gt = gt.unsqueeze(0)  # 添加批次维度
                gt = gt.float()  # 转换为浮点数类型
            train_data.append((torch.tensor([[p]], dtype=torch.float32).to(device), H.unsqueeze(0), None, None, gt))
    
    # 训练
    print(f"开始训练，共 {max_iter} 轮迭代")
    for it in range(max_iter):
        model.train()
        total_loss = 0.0
        
        # 监督损失
        for p_tensor, H_p, B_p, C_p, gt in train_data:
            pred = model(p_tensor)
            loss = criterion(pred, gt, H_p, B_p, C_p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 自适应采样更新
        if it % update_T == 0 and it > 0:
            new_col, fail_prob = adaptive_sampling(model, lambda p: generate_parametric_matrix(p,n), p_candidate)
            if len(new_col) > 0:
                p_col = np.concatenate([p_col, new_col])
        
        # 每1轮打印一次损失
        if it % 1 == 0:
            print(f'迭代 {it}/{max_iter}, 损失: {total_loss/num_train:.4f}')
    
    return model
def test_model(model, n=256, num_test=100, op='inv', equation_type='AX=B', generate_explanation=False):
    device = next(model.parameters()).device
    p_test = np.random.uniform(0, 1, num_test)
    rel_err = 0.0
    
    # 推理时间
    import time
    start = time.time()
    
    with torch.no_grad():
        for i, p in enumerate(p_test):
            p_tensor = torch.tensor([[p]], dtype=torch.float32).to(device)
            H = generate_parametric_matrix(p, n).to(device)
            pred = model(p_tensor)
            
            if equation_type == 'AX=B':
                # 生成随机B，然后使用智能方法求解X=A^{-1}B
                B = torch.randn(n, n, device=device)
                X = solve_linear_system(H, B, equation_type)
                # 测试AX=B的解
                err = torch.norm(torch.mm(H, pred.squeeze()) - B) / torch.norm(B)
                
                # 生成解释
                if generate_explanation and i == 0:  # 只对第一个测试案例生成解释
                    explanation = MatrixExplainer.generate_explanation(
                        equation_type, H.cpu(), B.cpu(), X=pred.squeeze().cpu()
                    )
                    print("\n=== 求解过程解释 ===")
                    print(explanation)
                    print("====================\n")
            elif equation_type == 'XA=B':
                # 生成随机B，然后使用智能方法求解X=BA^{-1}
                B = torch.randn(n, n, device=device)
                X = solve_linear_system(H.T, B.T, 'AX=B').T
                # 测试XA=B的解
                err = torch.norm(torch.mm(pred.squeeze(), H) - B) / torch.norm(B)
                
                # 生成解释
                if generate_explanation and i == 0:  # 只对第一个测试案例生成解释
                    explanation = MatrixExplainer.generate_explanation(
                        equation_type, H.cpu(), B.cpu(), X=pred.squeeze().cpu()
                    )
                    print("\n=== 求解过程解释 ===")
                    print(explanation)
                    print("====================\n")
            elif equation_type == 'AXB=C':
                # 生成随机C和B矩阵，然后使用智能方法求解X=A^{-1}CB^{-1}
                B_mat = generate_parametric_matrix(p + 0.1, n).to(device)
                C = torch.randn(n, n, device=device)
                Y = solve_linear_system(H, C, 'AX=B')
                X = solve_linear_system(B_mat.T, Y.T, 'AX=B').T
                # 测试AXB=C的解
                err = torch.norm(torch.mm(torch.mm(H, pred.squeeze()), B_mat) - C) / torch.norm(C)
                
                # 生成解释
                if generate_explanation and i == 0:  # 只对第一个测试案例生成解释
                    explanation = MatrixExplainer.generate_explanation(
                        equation_type, H.cpu(), B_mat.cpu(), C.cpu(), pred.squeeze().cpu()
                    )
                    print("\n=== 求解过程解释 ===")
                    print(explanation)
                    print("====================\n")
            else:  # 传统的求逆或SVD
                gt = get_ground_truth(p, n, op).to(device)
                if op == 'inv':
                    I = torch.eye(n).to(device)
                    err = torch.norm(H @ pred - I) / torch.norm(I)
                    
                    # 生成解释
                    if generate_explanation and i == 0:  # 只对第一个测试案例生成解释
                        explanation = MatrixExplainer.generate_explanation(
                            'inv', H.cpu(), X=pred.squeeze().cpu()
                        )
                        print("\n=== 求解过程解释 ===")
                        print(explanation)
                        print("====================\n")
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
    import time
    rel_err = 0.0
    start = time.time()
    for _ in range(num_test):
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
if __name__ == '__main__':
    # 超参数
    MATRIX_SIZE = 256
    OP_TYPE = 'inv'  # 'inv' 或 'svd'
    EQUATION_TYPE = 'AX=B'  # 'AX=B', 'XA=B', 'AXB=C' 或 'inv'
    GENERATE_EXPLANATION = True  # 是否生成求解过程解释
    
    # 训练
    print(f"开始训练 NeuMatC (方程类型: {EQUATION_TYPE})...")
    model = train_neumatc(n=MATRIX_SIZE, op=OP_TYPE, equation_type=EQUATION_TYPE)
    
    # 测试
    print("\nNeuMatC 测试:")
    test_model(model, n=MATRIX_SIZE, op=OP_TYPE, equation_type=EQUATION_TYPE, generate_explanation=GENERATE_EXPLANATION)
    
    print("\nNumPy基线测试:")
    baseline_test(n=MATRIX_SIZE, op=OP_TYPE)