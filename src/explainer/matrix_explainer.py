import torch
from ..utils.matrix_error_handler import MatrixErrorHandler
from ..utils.matrix_analyzer import MatrixAnalyzer

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
    
    @staticmethod
    def explain_solution(A, B, X, equation_type, matrix_info, method):
        """解释求解过程"""
        explanation = "# 矩阵方程求解解释\n\n"
        
        # 方程信息
        explanation += "## 1. 方程信息\n"
        explanation += MatrixExplainer.explain_equation(equation_type, A, B)
        explanation += f"\n求解方法: {method}\n"
        explanation += "\n"
        
        # 矩阵性质
        explanation += "## 2. 矩阵性质\n"
        for key, value in matrix_info.items():
            explanation += f"- {key}: {value}\n"
        explanation += "\n"
        
        # 求解步骤
        explanation += "## 3. 求解步骤\n"
        steps = MatrixExplainer.explain_solution_steps(equation_type, A, B, X=X)
        for i, step in enumerate(steps, 1):
            explanation += f"{i}. {step}\n"
        explanation += "\n"
        
        # 解的验证
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
        
        return explanation
    
    @staticmethod
    def explain_inverse(A, A_inv, matrix_info, method):
        """解释逆矩阵求解过程"""
        explanation = "# 逆矩阵求解解释\n\n"
        
        # 矩阵信息
        explanation += "## 1. 矩阵信息\n"
        explanation += f"矩阵A形状: {A.shape}\n"
        explanation += f"求解方法: {method}\n"
        explanation += "\n"
        
        # 矩阵性质
        explanation += "## 2. 矩阵性质\n"
        for key, value in matrix_info.items():
            explanation += f"- {key}: {value}\n"
        explanation += "\n"
        
        # 求解步骤
        explanation += "## 3. 求解步骤\n"
        steps = MatrixExplainer.explain_solution_steps('inv', A, X=A_inv)
        for i, step in enumerate(steps, 1):
            explanation += f"{i}. {step}\n"
        explanation += "\n"
        
        # 解的验证
        explanation += "## 4. 解的验证\n"
        AA_inv = A @ A_inv
        I = torch.eye(A.shape[0])
        error = torch.norm(AA_inv - I).item()
        explanation += f"计算AA^{-1}: 形状 = {AA_inv.shape}\n"
        explanation += f"验证误差: ||AA^{-1} - I|| = {error:.4e}\n"
        
        return explanation
    
    @staticmethod
    def explain_eigenvalues(A, eigenvalues, eigenvectors, matrix_info):
        """解释特征值计算过程"""
        explanation = "# 特征值计算解释\n\n"
        
        # 矩阵信息
        explanation += "## 1. 矩阵信息\n"
        explanation += f"矩阵A形状: {A.shape}\n"
        explanation += "\n"
        
        # 矩阵性质
        explanation += "## 2. 矩阵性质\n"
        for key, value in matrix_info.items():
            explanation += f"- {key}: {value}\n"
        explanation += "\n"
        
        # 特征值和特征向量
        explanation += "## 3. 特征值和特征向量\n"
        explanation += "### 特征值\n"
        for i, val in enumerate(eigenvalues):
            explanation += f"λ{i+1} = {val.item():.4f}\n"
        explanation += "\n"
        
        explanation += "### 特征向量\n"
        for i in range(eigenvectors.shape[1]):
            explanation += f"v{i+1} = {eigenvectors[:, i].numpy()}\n"
        explanation += "\n"
        
        # 验证
        explanation += "## 4. 验证\n"
        for i in range(len(eigenvalues)):
            lambda_i = eigenvalues[i]
            v_i = eigenvectors[:, i]
            Av = A @ v_i
            lambda_v = lambda_i * v_i
            error = torch.norm(Av - lambda_v).item()
            explanation += f"对于λ{i+1} = {lambda_i.item():.4f}: ||Av - λv|| = {error:.4e}\n"
        
        return explanation
    
    @staticmethod
    def explain_svd(A, U, S, Vt, matrix_info):
        """解释SVD分解过程"""
        explanation = "# SVD分解解释\n\n"
        
        # 矩阵信息
        explanation += "## 1. 矩阵信息\n"
        explanation += f"矩阵A形状: {A.shape}\n"
        explanation += "\n"
        
        # 矩阵性质
        explanation += "## 2. 矩阵性质\n"
        for key, value in matrix_info.items():
            explanation += f"- {key}: {value}\n"
        explanation += "\n"
        
        # SVD结果
        explanation += "## 3. SVD分解结果\n"
        explanation += f"U矩阵形状: {U.shape}\n"
        explanation += f"奇异值数量: {len(S)}\n"
        explanation += f"Vt矩阵形状: {Vt.shape}\n"
        explanation += "\n"
        
        explanation += "### 奇异值\n"
        for i, s in enumerate(S):
            explanation += f"σ{i+1} = {s.item():.4f}\n"
        explanation += "\n"
        
        # 验证
        explanation += "## 4. 验证\n"
        recon = U @ torch.diag_embed(S) @ Vt
        error = torch.norm(recon - A).item()
        explanation += f"重构误差: ||A - UΣV^T|| = {error:.4e}\n"
        
        # 正交性验证
        UUt = U @ U.T
        I = torch.eye(U.shape[0])
        error_U = torch.norm(UUt - I).item()
        explanation += f"U的正交性: ||UU^T - I|| = {error_U:.4e}\n"
        
        VtV = Vt @ Vt.T
        I = torch.eye(Vt.shape[0])
        error_V = torch.norm(VtV - I).item()
        explanation += f"V的正交性: ||V^TV - I|| = {error_V:.4e}\n"
        
        return explanation
    
    @staticmethod
    def explain_pca(X, X_pca, eigenvalues, eigenvectors, mean, n_components):
        """解释PCA降维过程"""
        explanation = "# PCA降维解释\n\n"
        
        # 数据信息
        explanation += "## 1. 数据信息\n"
        explanation += f"原始数据形状: {X.shape}\n"
        explanation += f"降维后数据形状: {X_pca.shape}\n"
        explanation += f"主成分数量: {n_components}\n"
        explanation += "\n"
        
        # 特征值
        explanation += "## 2. 特征值\n"
        total_var = sum(eigenvalues)
        for i, val in enumerate(eigenvalues):
            variance_ratio = val / total_var
            explanation += f"特征值{i+1}: {val.item():.4f} (方差占比: {variance_ratio:.4f})\n"
        explanation += "\n"
        
        # 主成分
        explanation += "## 3. 主成分\n"
        for i in range(eigenvectors.shape[1]):
            explanation += f"主成分{i+1}: {eigenvectors[:, i].numpy()}\n"
        explanation += "\n"
        
        # 均值
        explanation += "## 4. 均值\n"
        explanation += f"数据均值: {mean.numpy()}\n"
        
        return explanation