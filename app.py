import streamlit as st
import torch
import numpy as np
from src.solver.trainer import train_neumatc, test_model
from src.explainer.matrix_explainer import MatrixExplainer
from src.solver.algebraic_solver import solve_linear_system
from src.utils.data_generator import generate_parametric_matrix

# 设置页面标题
st.title('AI矩阵方程求解器')

# 侧边栏设置
st.sidebar.header('参数设置')
matrix_size = st.sidebar.slider('矩阵大小', min_value=2, max_value=100, value=10, step=1)
equation_type = st.sidebar.selectbox('方程类型', ['AX=B', 'XA=B', 'AXB=C', 'inv'])
op_type = st.sidebar.selectbox('操作类型', ['inv', 'svd'])
generate_explanation = st.sidebar.checkbox('生成求解过程解释', value=True)

# 主内容区域
st.header('矩阵方程求解')

# 生成随机矩阵
if st.button('生成随机矩阵并求解'):
    # 生成参数
    p = np.random.uniform(0, 1)
    
    # 生成矩阵
    A = generate_parametric_matrix(p, n=matrix_size)
    
    # 显示矩阵A
    st.subheader('矩阵A')
    st.write(A.numpy())
    
    # 根据方程类型生成其他矩阵
    if equation_type == 'AX=B':
        # 生成随机B
        B = torch.randn(matrix_size, matrix_size)
        st.subheader('矩阵B')
        st.write(B.numpy())
        
        # 求解X
        X = solve_linear_system(A, B, equation_type)
        st.subheader('解X')
        st.write(X.numpy())
        
        # 验证解
        AX = A @ X
        error = torch.norm(AX - B).item()
        st.subheader('验证结果')
        st.write(f'||AX - B|| = {error:.4e}')
        
        # 生成解释
        if generate_explanation:
            st.subheader('求解过程解释')
            explanation = MatrixExplainer.generate_explanation(
                equation_type, A, B, X=X
            )
            st.markdown(explanation)
    
    elif equation_type == 'XA=B':
        # 生成随机B
        B = torch.randn(matrix_size, matrix_size)
        st.subheader('矩阵B')
        st.write(B.numpy())
        
        # 求解X
        X = solve_linear_system(A.T, B.T, 'AX=B').T
        st.subheader('解X')
        st.write(X.numpy())
        
        # 验证解
        XA = X @ A
        error = torch.norm(XA - B).item()
        st.subheader('验证结果')
        st.write(f'||XA - B|| = {error:.4e}')
        
        # 生成解释
        if generate_explanation:
            st.subheader('求解过程解释')
            explanation = MatrixExplainer.generate_explanation(
                equation_type, A, B, X=X
            )
            st.markdown(explanation)
    
    elif equation_type == 'AXB=C':
        # 生成随机B和C
        B_mat = generate_parametric_matrix(p + 0.1, n=matrix_size)
        C = torch.randn(matrix_size, matrix_size)
        st.subheader('矩阵B')
        st.write(B_mat.numpy())
        st.subheader('矩阵C')
        st.write(C.numpy())
        
        # 求解X
        Y = solve_linear_system(A, C, 'AX=B')
        X = solve_linear_system(B_mat.T, Y.T, 'AX=B').T
        st.subheader('解X')
        st.write(X.numpy())
        
        # 验证解
        AXB = A @ X @ B_mat
        error = torch.norm(AXB - C).item()
        st.subheader('验证结果')
        st.write(f'||AXB - C|| = {error:.4e}')
        
        # 生成解释
        if generate_explanation:
            st.subheader('求解过程解释')
            explanation = MatrixExplainer.generate_explanation(
                equation_type, A, B_mat, C, X=X
            )
            st.markdown(explanation)
    
    elif equation_type == 'inv':
        # 求解A的逆
        X = solve_linear_system(A, torch.eye(matrix_size), 'AX=B')
        st.subheader('A的逆矩阵')
        st.write(X.numpy())
        
        # 验证解
        AA_inv = A @ X
        error = torch.norm(AA_inv - torch.eye(matrix_size)).item()
        st.subheader('验证结果')
        st.write(f'||AA^{-1} - I|| = {error:.4e}')
        
        # 生成解释
        if generate_explanation:
            st.subheader('求解过程解释')
            explanation = MatrixExplainer.generate_explanation(
                'inv', A, X=X
            )
            st.markdown(explanation)

# 模型训练部分
st.header('NeuMatC模型训练')
train_model = st.checkbox('训练模型')

if train_model:
    # 训练参数
    num_train = st.slider('训练样本数', min_value=10, max_value=100, value=40, step=10)
    max_iter = st.slider('训练迭代次数', min_value=1000, max_value=10000, value=5000, step=1000)
    
    if st.button('开始训练'):
        # 训练模型
        st.write('开始训练模型...')
        model = train_neumatc(n=matrix_size, op=op_type, equation_type=equation_type, num_train=num_train, max_iter=max_iter)
        
        # 测试模型
        st.write('测试模型...')
        rel_err, infer_time = test_model(model, n=matrix_size, op=op_type, equation_type=equation_type, generate_explanation=generate_explanation)
        
        # 显示结果
        st.subheader('模型测试结果')
        st.write(f'相对误差: {rel_err:.4e}')
        st.write(f'单矩阵推理时间: {infer_time:.2f} ms')