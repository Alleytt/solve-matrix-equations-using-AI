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