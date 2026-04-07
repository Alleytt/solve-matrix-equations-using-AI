# train_model.py
import torch
import matrix_equation

# 训练参数
MATRIX_SIZE = 16  # 矩阵大小
OP_TYPE = 'inv'  # 操作类型
NUM_TRAIN = 20  # 训练样本数
MAX_ITER = 1000  # 训练迭代次数

print("开始训练 NeuMatC 模型...")
print(f"矩阵大小: {MATRIX_SIZE}")
print(f"操作类型: {OP_TYPE}")
print(f"训练样本数: {NUM_TRAIN}")
print(f"训练迭代次数: {MAX_ITER}")

# 训练模型
model = matrix_equation.train_neumatc(n=MATRIX_SIZE, op=OP_TYPE, num_train=NUM_TRAIN, max_iter=MAX_ITER)

# 保存模型权重
torch.save(model.state_dict(), 'model.pth')
print("\n模型训练完成并保存为 'model.pth'")