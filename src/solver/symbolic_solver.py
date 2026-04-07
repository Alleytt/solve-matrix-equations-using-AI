import sympy as sp
import torch
import numpy as np

class SymbolicSolver:
    """符号矩阵方程求解器"""
    
    @staticmethod
    def solve_ax_b(A, B):
        """求解符号线性方程组 AX=B"""
        # 将PyTorch张量转换为SymPy矩阵
        if isinstance(A, torch.Tensor):
            A = sp.Matrix(A.cpu().numpy())
        elif isinstance(A, np.ndarray):
            A = sp.Matrix(A)
        
        if isinstance(B, torch.Tensor):
            B = sp.Matrix(B.cpu().numpy())
        elif isinstance(B, np.ndarray):
            B = sp.Matrix(B)
        
        # 检查A是否可逆
        if A.det() == 0:
            raise ValueError("矩阵A是奇异的，无法直接求解")
        
        # 求解X = A^{-1}B
        X = A.inv() * B
        return X
    
    @staticmethod
    def solve_xa_b(A, B):
        """求解符号线性方程组 XA=B"""
        # 将PyTorch张量转换为SymPy矩阵
        if isinstance(A, torch.Tensor):
            A = sp.Matrix(A.cpu().numpy())
        elif isinstance(A, np.ndarray):
            A = sp.Matrix(A)
        
        if isinstance(B, torch.Tensor):
            B = sp.Matrix(B.cpu().numpy())
        elif isinstance(B, np.ndarray):
            B = sp.Matrix(B)
        
        # 检查A是否可逆
        if A.det() == 0:
            raise ValueError("矩阵A是奇异的，无法直接求解")
        
        # 求解X = BA^{-1}
        X = B * A.inv()
        return X
    
    @staticmethod
    def solve_axb_c(A, B, C):
        """求解符号线性方程组 AXB=C"""
        # 将PyTorch张量转换为SymPy矩阵
        if isinstance(A, torch.Tensor):
            A = sp.Matrix(A.cpu().numpy())
        elif isinstance(A, np.ndarray):
            A = sp.Matrix(A)
        
        if isinstance(B, torch.Tensor):
            B = sp.Matrix(B.cpu().numpy())
        elif isinstance(B, np.ndarray):
            B = sp.Matrix(B)
        
        if isinstance(C, torch.Tensor):
            C = sp.Matrix(C.cpu().numpy())
        elif isinstance(C, np.ndarray):
            C = sp.Matrix(C)
        
        # 检查A和B是否可逆
        if A.det() == 0:
            raise ValueError("矩阵A是奇异的，无法直接求解")
        if B.det() == 0:
            raise ValueError("矩阵B是奇异的，无法直接求解")
        
        # 求解X = A^{-1}CB^{-1}
        X = A.inv() * C * B.inv()
        return X
    
    @staticmethod
    def solve_matrix_inverse(A):
        """求解符号矩阵的逆"""
        # 将PyTorch张量转换为SymPy矩阵
        if isinstance(A, torch.Tensor):
            A = sp.Matrix(A.cpu().numpy())
        elif isinstance(A, np.ndarray):
            A = sp.Matrix(A)
        
        # 检查A是否可逆
        if A.det() == 0:
            raise ValueError("矩阵A是奇异的，无法求逆")
        
        # 求解A的逆
        A_inv = A.inv()
        return A_inv
    
    @staticmethod
    def simplify_expression(expr):
        """简化符号表达式"""
        return sp.simplify(expr)
    
    @staticmethod
    def evaluate_expression(expr, values):
        """代入数值计算符号表达式"""
        return expr.subs(values)
    
    @staticmethod
    def create_symbolic_matrix(shape, symbols=None):
        """创建符号矩阵"""
        if symbols is None:
            # 自动生成符号
            symbols = []
            for i in range(shape[0]):
                row = []
                for j in range(shape[1]):
                    row.append(sp.Symbol(f'a_{i}{j}'))
                symbols.append(row)
        return sp.Matrix(symbols)