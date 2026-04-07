import sympy as sp
from src.solver.symbolic_solver import SymbolicSolver

# 示例1: 求解简单的符号线性方程组 AX=B
print("示例1: 求解符号线性方程组 AX=B")

# 创建符号矩阵A和B
A = sp.Matrix([[sp.Symbol('a'), sp.Symbol('b')], 
               [sp.Symbol('c'), sp.Symbol('d')]])
B = sp.Matrix([[sp.Symbol('e')], 
               [sp.Symbol('f')]])

# 求解X
X = SymbolicSolver.solve_ax_b(A, B)
print("解X:")
print(X)
print()

# 示例2: 求解矩阵逆
print("示例2: 求解符号矩阵的逆")

# 创建符号矩阵A
A = sp.Matrix([[sp.Symbol('a'), sp.Symbol('b')], 
               [sp.Symbol('c'), sp.Symbol('d')]])

# 求解A的逆
A_inv = SymbolicSolver.solve_matrix_inverse(A)
print("A的逆:")
print(A_inv)
print()

# 示例3: 代入数值计算
print("示例3: 代入数值计算")

# 定义数值
values = {sp.Symbol('a'): 1, sp.Symbol('b'): 2, 
          sp.Symbol('c'): 3, sp.Symbol('d'): 4, 
          sp.Symbol('e'): 5, sp.Symbol('f'): 6}

# 代入数值计算X
X_numeric = X.subs(values)
print("代入数值后的X:")
print(X_numeric)
print()

# 示例4: 求解AXB=C
print("示例4: 求解符号线性方程组 AXB=C")

# 创建符号矩阵A、B和C
A = sp.Matrix([[sp.Symbol('a'), sp.Symbol('b')], 
               [sp.Symbol('c'), sp.Symbol('d')]])
B = sp.Matrix([[sp.Symbol('e'), sp.Symbol('f')], 
               [sp.Symbol('g'), sp.Symbol('h')]])
C = sp.Matrix([[sp.Symbol('i'), sp.Symbol('j')], 
               [sp.Symbol('k'), sp.Symbol('l')]])

# 求解X
X = SymbolicSolver.solve_axb_c(A, B, C)
print("解X:")
print(X)
print()