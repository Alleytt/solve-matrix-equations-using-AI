import torch
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    """求解器基类"""
    
    @abstractmethod
    def solve(self, A, B, **kwargs):
        """求解线性方程组 Ax = B"""
        pass
    
    def validate_input(self, A, B):
        """验证输入矩阵"""
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        if A.dim() != 2:
            raise ValueError("矩阵A必须是二维张量")
        
        if B.dim() not in [1, 2]:
            raise ValueError("矩阵B必须是一维或二维张量")
        
        if A.shape[0] != B.shape[0]:
            raise ValueError("矩阵A和B的行数必须相同")

class GaussianSolver(BaseSolver):
    """高斯消元法求解器"""
    
    def solve(self, A, B, use_pivoting=True, **kwargs):
        """使用高斯消元法求解线性方程组"""
        self.validate_input(A, B)
        
        # 转换为增广矩阵
        if B.dim() == 1:
            B = B.unsqueeze(1)
        
        aug_matrix = torch.cat([A, B], dim=1)
        n = A.shape[0]
        
        # 前向消元
        for i in range(n):
            # 部分选主元
            if use_pivoting:
                max_row = i + torch.argmax(torch.abs(aug_matrix[i:, i]))
                if max_row != i:
                    aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]
            
            # 归一化主行
            pivot = aug_matrix[i, i]
            if abs(pivot) < 1e-10:
                raise ValueError("矩阵是奇异的")
            
            aug_matrix[i] = aug_matrix[i] / pivot
            
            # 消去其他行
            for j in range(n):
                if j != i:
                    factor = aug_matrix[j, i]
                    aug_matrix[j] = aug_matrix[j] - factor * aug_matrix[i]
        
        # 提取解
        X = aug_matrix[:, n:]
        return X.squeeze() if B.shape[1] == 1 else X

class LUSolver(BaseSolver):
    """LU分解求解器"""
    
    def solve(self, A, B, **kwargs):
        """使用LU分解求解线性方程组"""
        self.validate_input(A, B)
        
        try:
            LU, pivots = torch.linalg.lu_factor(A)
            return torch.linalg.lu_solve(LU, pivots, B)
        except Exception as e:
            raise ValueError(f"LU分解失败: {str(e)}")

class QRSolver(BaseSolver):
    """QR分解求解器"""
    
    def solve(self, A, B, **kwargs):
        """使用QR分解求解线性方程组"""
        self.validate_input(A, B)
        
        try:
            Q, R = torch.linalg.qr(A)
            return torch.linalg.solve(R, Q.T @ B)
        except Exception as e:
            raise ValueError(f"QR分解失败: {str(e)}")

class SVDSolver(BaseSolver):
    """SVD分解求解器"""
    
    def solve(self, A, B, **kwargs):
        """使用SVD分解求解线性方程组"""
        self.validate_input(A, B)
        
        try:
            U, S, Vt = torch.linalg.svd(A)
            S_inv = torch.diag_embed(1.0 / S)
            return Vt.T @ S_inv @ U.T @ B
        except Exception as e:
            raise ValueError(f"SVD分解失败: {str(e)}")

class CholeskySolver(BaseSolver):
    """Cholesky分解求解器"""
    
    def solve(self, A, B, **kwargs):
        """使用Cholesky分解求解线性方程组"""
        self.validate_input(A, B)
        
        try:
            L = torch.linalg.cholesky(A)
            return torch.cholesky_solve(B, L)
        except Exception as e:
            raise ValueError(f"Cholesky分解失败: {str(e)}")

class PseudoInverseSolver(BaseSolver):
    """伪逆求解器"""
    
    def solve(self, A, B, **kwargs):
        """使用伪逆求解线性方程组"""
        self.validate_input(A, B)
        
        try:
            return torch.linalg.pinv(A) @ B
        except Exception as e:
            raise ValueError(f"伪逆计算失败: {str(e)}")
