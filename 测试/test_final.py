print("开始测试")

try:
    import matrix_equation
    
    # 测试参数
    MATRIX_SIZE = 8  # 减小矩阵大小以加快测试速度
    OP_TYPE = 'inv'  # 'inv' 或 'svd'
    NUM_TRAIN = 2  # 减少训练样本数
    MAX_ITER = 3  # 减少训练迭代次数
    
    print("开始测试 train_neumatc 函数")
    print(f"矩阵大小: {MATRIX_SIZE}")
    print(f"操作类型: {OP_TYPE}")
    print(f"训练样本数: {NUM_TRAIN}")
    print(f"训练迭代次数: {MAX_ITER}")
    print("\n开始训练...")
    
    # 调用 train_neumatc 函数
    model = matrix_equation.train_neumatc(n=MATRIX_SIZE, op=OP_TYPE, num_train=NUM_TRAIN, max_iter=MAX_ITER)
    
    print("\n训练完成，开始测试")
    
    # 测试模型
    print("\nNeuMatC 测试:")
    matrix_equation.test_model(model, n=MATRIX_SIZE, op=OP_TYPE, num_test=2)
    
    # 基线测试
    print("\nNumPy基线测试:")
    matrix_equation.baseline_test(n=MATRIX_SIZE, op=OP_TYPE, num_test=2)
    
    print("\n测试完成")
    
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()
