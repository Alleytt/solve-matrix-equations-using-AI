print("开始测试模块函数")

import matrix_equation
import torch
import numpy as np

print("测试 1: generate_parametric_matrix 函数")
try:
    H = matrix_equation.generate_parametric_matrix(0.5, n=4)
    print(f"  生成矩阵形状: {H.shape}")
    print("  测试成功")
except Exception as e:
    print(f"  测试失败: {e}")

print("\n测试 2: get_ground_truth 函数")
try:
    gt = matrix_equation.get_ground_truth(0.5, n=4, op='inv')
    print(f"  真实值形状: {gt.shape}")
    print("  测试成功")
except Exception as e:
    print(f"  测试失败: {e}")

print("\n测试 3: LowRankContinuousMapping 类")
try:
    model = matrix_equation.LowRankContinuousMapping(input_dim=1, hidden_dim=10, latent_dim=5, 
                                                   output_shape=(4, 4), activation='sin')
    print("  模型创建成功")
    # 测试前向传播
    p_tensor = torch.tensor([[0.5]], dtype=torch.float32)
    pred = model(p_tensor)
    print(f"  模型输出形状: {pred.shape}")
    print("  测试成功")
except Exception as e:
    print(f"  测试失败: {e}")

print("\n测试 4: AlgebraicLoss 类")
try:
    criterion = matrix_equation.AlgebraicLoss(op_type='inv', lambda_consist=1.0)
    print("  损失函数创建成功")
    # 测试损失计算
    H = matrix_equation.generate_parametric_matrix(0.5, n=4)
    gt = matrix_equation.get_ground_truth(0.5, n=4, op='inv').unsqueeze(0).float()
    model = matrix_equation.LowRankContinuousMapping(input_dim=1, hidden_dim=10, latent_dim=5, 
                                                   output_shape=(4, 4), activation='sin')
    p_tensor = torch.tensor([[0.5]], dtype=torch.float32)
    pred = model(p_tensor)
    loss = criterion(pred, gt, H.unsqueeze(0))
    print(f"  损失值: {loss.item()}")
    print("  测试成功")
except Exception as e:
    print(f"  测试失败: {e}")

print("\n测试 5: adaptive_sampling 函数")
try:
    model = matrix_equation.LowRankContinuousMapping(input_dim=1, hidden_dim=10, latent_dim=5, 
                                                   output_shape=(4, 4), activation='sin')
    candidate_p = np.random.uniform(0, 1, 10)
    new_col, fail_prob = matrix_equation.adaptive_sampling(model, lambda p: matrix_equation.generate_parametric_matrix(p, n=4), candidate_p)
    print(f"  新采样点数量: {len(new_col)}")
    print(f"  失败概率: {fail_prob}")
    print("  测试成功")
except Exception as e:
    print(f"  测试失败: {e}")

print("\n测试完成")
