import torch
import matrix_equation

# 测试参数
MATRIX_SIZE = 32
OP_TYPE = 'inv'

# 创建模型
model = matrix_equation.LowRankContinuousMapping(input_dim=1, hidden_dim=100, latent_dim=20, 
                                                 output_shape=(MATRIX_SIZE, MATRIX_SIZE), activation='sin')

# 测试前向传播
p_tensor = torch.tensor([[0.5]], dtype=torch.float32)
pred = model(p_tensor)
print(f"模型输出形状: {pred.shape}")
print("前向传播测试成功")

# 测试生成矩阵
H = matrix_equation.generate_parametric_matrix(0.5, n=MATRIX_SIZE)
print(f"生成矩阵形状: {H.shape}")
print("生成矩阵测试成功")

# 测试获取真实值
gt = matrix_equation.get_ground_truth(0.5, n=MATRIX_SIZE, op=OP_TYPE)
print(f"真实值形状: {gt.shape}")
print("获取真实值测试成功")
