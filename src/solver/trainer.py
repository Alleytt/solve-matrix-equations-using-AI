import torch
import torch.optim as optim
import numpy as np
from ..models.low_rank_continuous_mapping import LowRankContinuousMapping
from ..solver.algebraic_solver import AlgebraicLoss, solve_linear_system
from ..utils.adaptive_sampling import adaptive_sampling
from ..utils.data_generator import generate_parametric_matrix, get_ground_truth
from ..utils.matrix_error_handler import MatrixErrorHandler
from ..explainer.matrix_explainer import MatrixExplainer

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