import torch
from .base_solver import (
    GaussianSolver, LUSolver, QRSolver, SVDSolver,
    CholeskySolver, PseudoInverseSolver
)
from ..utils.matrix_analyzer import MatrixAnalyzer
from ..utils.matrix_error_handler import MatrixErrorHandler

class IntelligentSolver:
    """智能求解器，根据矩阵特性自动选择最佳求解方法"""
    
    def __init__(self):
        self.solver_map = {
            "gaussian": GaussianSolver(),
            "lu": LUSolver(),
            "qr": QRSolver(),
            "svd": SVDSolver(),
            "cholesky": CholeskySolver(),
            "pinv": PseudoInverseSolver()
        }
    
    def select_solver(self, A, B, equation_type='AX=B'):
        """根据矩阵特性选择最佳求解方法"""
        n = A.shape[0]
        
        # 检查矩阵是否奇异
        if MatrixErrorHandler.is_singular(A):
            return "pinv"
        
        # 检查矩阵是否正定
        if MatrixAnalyzer.is_positive_definite(A):
            return "cholesky"
        
        # 检查矩阵是否病态
        if MatrixAnalyzer.is_ill_conditioned(A):
            return "svd"
        
        # 根据矩阵大小选择方法
        if n < 100:
            return "gaussian"  # 小矩阵使用高斯消元
        elif n < 1000:
            return "lu"  # 中等矩阵使用LU分解
        else:
            return "qr"  # 大矩阵使用QR分解
    
    def solve(self, A, B, equation_type='AX=B', method="auto", **kwargs):
        """求解线性方程组"""
        # 处理不同类型的方程
        if equation_type == 'XA=B':
            # XA=B -> A^T X^T = B^T
            X = self.solve(A.T, B.T, 'AX=B', method, **kwargs)
            return X.T
        elif equation_type == 'AXB=C':
            # AXB=C -> 先求解AY=C，再求解 Y=XB
            Y = self.solve(A, B, 'AX=B', method, **kwargs)
            X = self.solve(B.T, Y.T, 'AX=B', method, **kwargs)
            return X.T
        
        # 选择求解方法
        if method == "auto":
            solver_name = self.select_solver(A, B, equation_type)
        else:
            solver_name = method
        
        print(f"使用求解方法: {solver_name}")
        
        # 获取对应的求解器
        solver = self.solver_map.get(solver_name)
        if not solver:
            raise ValueError(f"未知的求解方法: {solver_name}")
        
        # 尝试使用选择的方法求解
        try:
            return solver.solve(A, B, **kwargs)
        except Exception as e:
            print(f"{solver_name} 方法失败: {str(e)}")
            # 降级使用其他方法
            if solver_name != "pinv":
                print("降级使用伪逆方法")
                return self.solver_map["pinv"].solve(A, B, **kwargs)
            raise
    
    def solve_inverse(self, A, method="auto", **kwargs):
        """求解矩阵的逆"""
        n = A.shape[0]
        I = torch.eye(n, device=A.device)
        return self.solve(A, I, 'AX=B', method, **kwargs)
    
    def solve_eigenvalues(self, A):
        """计算矩阵的特征值"""
        try:
            eigenvalues, eigenvectors = torch.linalg.eig(A)
            return eigenvalues, eigenvectors
        except Exception as e:
            raise ValueError(f"特征值计算失败: {str(e)}")
    
    def solve_svd(self, A):
        """计算矩阵的SVD分解"""
        try:
            U, S, Vt = torch.linalg.svd(A)
            return U, S, Vt
        except Exception as e:
            raise ValueError(f"SVD分解失败: {str(e)}")
    
    def solve_pca(self, X, n_components=None):
        """执行PCA降维"""
        # 中心化数据
        mean = torch.mean(X, dim=0)
        X_centered = X - mean
        
        # 计算协方差矩阵
        covariance = torch.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        
        # 按特征值降序排序
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 选择前n_components个主成分
        if n_components is not None:
            eigenvectors = eigenvectors[:, :n_components]
        
        # 投影数据
        X_pca = torch.matmul(X_centered, eigenvectors)
        
        return X_pca, eigenvalues, eigenvectors, mean
