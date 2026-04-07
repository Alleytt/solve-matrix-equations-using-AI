import matrix_equation

# 测试参数
MATRIX_SIZE = 32  # 减小矩阵大小以加快测试速度
OP_TYPE = 'inv'  # 'inv' 或 'svd'
NUM_TRAIN = 10  # 减少训练样本数
MAX_ITER = 1000  # 减少训练迭代次数

print("开始训练 NeuMatC...")
model = matrix_equation.train_neumatc(n=MATRIX_SIZE, op=OP_TYPE, num_train=NUM_TRAIN, max_iter=MAX_ITER)

print("\nNeuMatC 测试:")
matrix_equation.test_model(model, n=MATRIX_SIZE, op=OP_TYPE, num_test=10)

print("\nNumPy基线测试:")
matrix_equation.baseline_test(n=MATRIX_SIZE, op=OP_TYPE, num_test=10)
