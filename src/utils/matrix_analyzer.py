import torch
from .matrix_error_handler import MatrixErrorHandler

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