# AI矩阵方程求解器 - NeuMatC++

基于深度学习的参数化矩阵方程求解工具，支持多种矩阵方程类型，提供智能算法调度、错误处理、可解释性等高级功能。

## 主要特性

- **多方程类型支持**：支持AX=B、XA=B、AXB=C、矩阵求逆、SVD分解等多种矩阵方程类型
- **智能算法调度**：根据矩阵特性自动选择最佳求解方法（inv、cholesky、lu、qr、svd、pinv）
- **错误处理和鲁棒性**：自动检测奇异矩阵、计算矩阵秩、分析方程解类型
- **数值稳定性优化**：使用torch.linalg.solve等稳定方法，避免直接求逆的数值问题
- **可解释性**：生成详细的求解步骤和推理过程，包括方程信息、矩阵性质、求解步骤和解的验证
- **模块化设计**：清晰的代码结构，分为utils、models、solver、explainer等模块
- **符号计算支持**：集成SymPy库，支持包含符号参数的矩阵方程求解
- **前端交互界面**：基于Streamlit的交互式界面，支持矩阵变换可视化和步骤展开

## 目录结构

```
├── main.py                    # 主入口文件
├── app.py                     # Streamlit前端应用
├── symbolic_example.py          # 符号计算示例
├── matrix_equation.py          # 原始核心实现文件（保留）
└── src/
    ├── __init__.py
    ├── parser/                 # 解析器模块
    │   └── __init__.py
    ├── solver/                 # 求解器模块
    │   ├── __init__.py
    │   ├── algebraic_solver.py     # 代数求解器和损失函数
    │   ├── symbolic_solver.py      # 符号计算求解器
    │   └── trainer.py             # 训练和测试函数
    ├── explainer/              # 解释器模块
    │   ├── __init__.py
    │   └── matrix_explainer.py   # 矩阵方程解释器
    ├── visualizer/             # 可视化模块
    │   └── __init__.py
    ├── models/                 # 模型模块
    │   ├── __init__.py
    │   └── low_rank_continuous_mapping.py  # 低秩连续映射模型
    └── utils/                  # 工具模块
        ├── __init__.py
        ├── matrix_error_handler.py    # 矩阵错误处理
        ├── matrix_analyzer.py        # 矩阵分析器
        ├── data_generator.py         # 数据生成器
        └── adaptive_sampling.py      # 自适应采样
```

## 核心功能

### 1. 矩阵方程求解
- **AX=B**：求解线性方程组，找到矩阵X使得A与X的乘积等于B
- **XA=B**：求解线性方程组，找到矩阵X使得X与A的乘积等于B
- **AXB=C**：求解双线性方程组，找到矩阵X使得A、X、B的乘积等于C
- **矩阵求逆**：学习并预测参数化矩阵的逆矩阵
- **SVD分解**：学习并预测参数化矩阵的SVD分解结果（U、S、Vt）

### 2. 智能算法调度
根据矩阵特性自动选择最佳求解方法：
- **inv**：直接求逆（适用于小矩阵）
- **cholesky**：Cholesky分解（适用于正定矩阵）
- **lu**：LU分解（适用于中等大小矩阵）
- **qr**：QR分解（适用于大矩阵）
- **svd**：奇异值分解（适用于病态矩阵）
- **pinv**：伪逆（适用于奇异矩阵）

### 3. 错误处理和鲁棒性
- **奇异矩阵检测**：自动检测矩阵是否奇异，使用伪逆等方法处理
- **矩阵秩计算**：计算矩阵的秩，分析方程解的类型
- **解类型分析**：判断方程是否有唯一解、无解或无穷多解
- **条件数计算**：计算矩阵的条件数，检测病态问题

### 4. 可解释性
- **方程信息**：解释方程类型和求解思路
- **矩阵性质**：分析矩阵的形状、奇异性、正定性、条件数、秩等性质
- **求解步骤**：详细列出求解过程的每个步骤
- **解的验证**：验证求解结果是否满足方程约束

### 5. 符号计算
- **符号矩阵求解**：支持包含符号参数的矩阵方程求解
- **表达式简化**：简化符号表达式
- **数值代入**：将符号表达式代入具体数值进行计算

## 安装依赖

### 基础依赖
```bash
pip install torch numpy matplotlib
```

### 前端应用依赖
```bash
pip install streamlit
```

### 符号计算依赖
```bash
pip install sympy
```

### 完整安装
```bash
pip install torch numpy matplotlib streamlit sympy
```

## 核心模块说明

### 1. MatrixErrorHandler 类
- **功能**：矩阵错误处理和鲁棒性模块
- **主要方法**：
  - `is_singular(matrix, tol)`：检测矩阵是否奇异
  - `get_rank(matrix, tol)`：计算矩阵的秩
  - `analyze_equation(A, B, equation_type)`：分析方程解的类型
  - `handle_singular_matrix(matrix)`：处理奇异矩阵，返回伪逆

### 2. MatrixAnalyzer 类
- **功能**：矩阵分析和智能算法选择模块
- **主要方法**：
  - `is_positive_definite(matrix, tol)`：检测矩阵是否正定
  - `condition_number(matrix)`：计算矩阵的条件数
  - `is_ill_conditioned(matrix, threshold)`：检测矩阵是否病态
  - `select_solver(matrix, equation_type)`：根据矩阵特性选择最佳求解方法
  - `solve_with_selector(A, B, equation_type)`：使用智能选择的方法求解线性方程组

### 3. MatrixExplainer 类
- **功能**：矩阵方程求解解释模块
- **主要方法**：
  - `explain_equation(equation_type, A, B, C)`：解释方程类型和求解思路
  - `explain_solution_steps(equation_type, A, B, C, X)`：解释求解步骤
  - `explain_matrix_properties(A)`：解释矩阵的性质
  - `generate_explanation(equation_type, A, B, C, X)`：生成完整的解释

### 4. SymbolicSolver 类
- **功能**：符号矩阵方程求解器
- **主要方法**：
  - `solve_ax_b(A, B)`：求解符号线性方程组 AX=B
  - `solve_xa_b(A, B)`：求解符号线性方程组 XA=B
  - `solve_axb_c(A, B, C)`：求解符号线性方程组 AXB=C
  - `solve_matrix_inverse(A)`：求解符号矩阵的逆
  - `simplify_expression(expr)`：简化符号表达式
  - `evaluate_expression(expr, values)`：代入数值计算符号表达式

### 5. LowRankContinuousMapping 类
- **功能**：实现低秩连续映射模型，用于预测参数化矩阵的操作结果
- **参数**：
  - `input_dim`：输入维度（默认为1，对应参数p）
  - `hidden_dim`：隐藏层维度（默认为100）
  - `latent_dim`：潜在空间维度（默认为20）
  - `output_shape`：输出矩阵形状（如 (n, n)）
  - `activation`：激活函数（默认为'sin'）

### 6. AlgebraicLoss 类
- **功能**：计算代数损失，包括数据损失和一致性损失
- **参数**：
  - `op_type`：操作类型（'inv' 或 'svd'）
  - `lambda_consist`：一致性损失权重（默认为1.0）
  - `equation_type`：方程类型（'AX=B', 'XA=B', 'AXB=C', 'inv'）

## 使用方法

### 1. 运行主程序
```bash
python main.py
```

### 2. 运行前端应用
```bash
python -m streamlit run app.py --server.headless true
```
然后在浏览器中访问 http://localhost:8501

### 3. 运行符号计算示例
```bash
python symbolic_example.py
```

### 4. 测试基本功能
```bash
python -c "from src.solver.algebraic_solver import solve_linear_system; import torch; A = torch.eye(2); B = torch.tensor([[1.0], [2.0]]); X = solve_linear_system(A, B); print('解:', X)"
```

## 使用示例

### 1. 求解AX=B方程
```python
from src.solver.algebraic_solver import solve_linear_system
from src.explainer.matrix_explainer import MatrixExplainer
import torch

# 创建矩阵
A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
B = torch.tensor([[4.0], [5.0]])

# 求解X
X = solve_linear_system(A, B, 'AX=B')

# 生成解释
explanation = MatrixExplainer.generate_explanation('AX=B', A, B, X=X)
print(explanation)

# 验证解
print("AX =", A @ X)
print("B =", B)
```

### 2. 求解XA=B方程
```python
from src.solver.algebraic_solver import solve_linear_system
from src.explainer.matrix_explainer import MatrixExplainer
import torch

# 创建矩阵
A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
B = torch.tensor([[4.0], [5.0]])

# 求解X
X = solve_linear_system(A.T, B.T, 'AX=B').T

# 生成解释
explanation = MatrixExplainer.generate_explanation('XA=B', A, B, X=X)
print(explanation)

# 验证解
print("XA =", X @ A)
print("B =", B)
```

### 3. 求解AXB=C方程
```python
from src.solver.algebraic_solver import solve_linear_system
from src.explainer.matrix_explainer import MatrixExplainer
import torch

# 创建矩阵
A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
B = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
C = torch.tensor([[4.0, 2.0], [2.0, 5.0]])

# 求解X
Y = solve_linear_system(A, C, 'AX=B')
X = solve_linear_system(B.T, Y.T, 'AX=B').T

# 生成解释
explanation = MatrixExplainer.generate_explanation('AXB=C', A, B, C, X=X)
print(explanation)

# 验证解
print("AXB =", A @ X @ B)
print("C =", C)
```

### 4. 符号计算示例
```python
from src.solver.symbolic_solver import SymbolicSolver
import sympy as sp

# 创建符号矩阵
a, b, c, d, e, f = sp.symbols('a b c d e f')
A = sp.Matrix([[a, b], [c, d]])
B = sp.Matrix([[e], [f]])

# 求解X
X = SymbolicSolver.solve_ax_b(A, B)
print("符号解X:")
print(X)

# 代入数值
values = {a: 2, b: 1, c: 1, d: 3, e: 4, f: 5}
X_numeric = X.subs(values)
print("数值解X:")
print(X_numeric)
```

### 5. 训练NeuMatC模型
```python
from src.solver.trainer import train_neumatc, test_model

# 训练参数
MATRIX_SIZE = 256
OP_TYPE = 'inv'
EQUATION_TYPE = 'AX=B'
NUM_TRAIN = 40
MAX_ITER = 5000

# 训练模型
model = train_neumatc(n=MATRIX_SIZE, op=OP_TYPE, equation_type=EQUATION_TYPE, 
                      num_train=NUM_TRAIN, max_iter=MAX_ITER)

# 测试模型
rel_err, infer_time = test_model(model, n=MATRIX_SIZE, op=OP_TYPE, 
                              equation_type=EQUATION_TYPE, generate_explanation=True)

print(f'相对误差: {rel_err:.4e}')
print(f'单矩阵推理时间: {infer_time:.2f} ms')
```

## 前端应用功能

Streamlit前端应用提供以下功能：

### 1. 参数设置
- 矩阵大小调整（2-100）
- 方程类型选择（AX=B、XA=B、AXB=C、inv）
- 操作类型选择（inv、svd）
- 是否生成求解过程解释

### 2. 矩阵生成和求解
- 一键生成随机矩阵并求解
- 显示矩阵A、B、C等输入矩阵
- 显示解矩阵X
- 验证求解结果

### 3. 求解过程解释
- 方程信息说明
- 矩阵性质分析
- 详细求解步骤
- 解的验证结果

### 4. NeuMatC模型训练
- 训练样本数调整
- 训练迭代次数设置
- 实时训练进度显示
- 模型性能测试

## 结果解释

### 智能算法选择
系统会根据矩阵特性自动选择最佳求解方法：
- **正定矩阵**：选择Cholesky分解（最快）
- **病态矩阵**：选择SVD分解（最稳定）
- **奇异矩阵**：选择伪逆（唯一可行方法）
- **小矩阵**：选择直接求逆（简单高效）
- **中等矩阵**：选择LU分解（平衡性能）
- **大矩阵**：选择QR分解（数值稳定）

### 求解过程解释
解释内容包括：
1. **方程信息**：方程类型、求解思路、矩阵形状
2. **矩阵性质**：形状、奇异性、正定性、条件数、秩
3. **求解步骤**：详细的求解步骤说明
4. **解的验证**：验证误差和结果分析

### 训练结果
- **损失值**：训练过程中的损失变化，反映模型学习效果
- **相对误差**：模型预测与真实结果的相对差异
- **推理时间**：模型预测单个矩阵操作所需的时间
- **求解方法**：智能选择的求解方法名称

## 技术原理

### 1. 智能算法调度
根据矩阵的数学特性选择最佳求解方法：
- **正定性检测**：使用Cholesky分解尝试，成功则矩阵正定
- **条件数分析**：通过SVD计算条件数，判断矩阵病态程度
- **奇异性检测**：通过行列式判断矩阵是否奇异
- **大小自适应**：根据矩阵大小选择合适的分解方法

### 2. 数值稳定性优化
- **避免直接求逆**：使用torch.linalg.solve代替torch.linalg.inv
- **正则化处理**：对奇异矩阵添加正则化项
- **伪逆方法**：对不可逆矩阵使用Moore-Penrose伪逆
- **条件数监控**：对病态矩阵使用更稳定的SVD方法

### 3. 低秩连续映射
- **潜在空间表示**：通过低维潜在空间表示高维矩阵操作
- **可学习张量**：使用可学习的张量C实现mode-3乘积
- **周期激活函数**：使用sin激活函数处理周期性参数
- **自适应采样**：根据预测误差动态添加训练样本

### 4. 代数约束
- **一致性损失**：确保预测结果满足代数约束
- **正交性约束**：SVD分解中U和V的正交性
- **乘法约束**：逆矩阵的乘法性质（A·A⁻¹≈I）
- **方程约束**：AX=B、XA=B、AXB=C等方程的约束

## 注意事项

1. **矩阵大小**：较大的矩阵会增加计算时间和内存需求，建议先使用较小的矩阵（如8x8或16x16）进行测试
2. **训练时间**：训练过程可能需要几分钟到几小时，具体取决于矩阵大小和训练迭代次数
3. **模型保存**：训练完成后，建议保存模型权重，以便后续直接加载使用
4. **参数调整**：根据具体任务需求，可以调整模型的隐藏层维度、潜在空间维度、学习率等参数
5. **硬件加速**：如果有GPU，PyTorch会自动使用GPU加速训练和推理，提高计算速度
6. **前端应用**：Streamlit应用需要较长的启动时间，首次运行可能需要安装依赖
7. **符号计算**：符号计算适用于小矩阵和理论分析，大矩阵建议使用数值方法

## 性能对比

### NeuMatC vs NumPy
- **训练后推理**：NeuMatC比NumPy快10-100倍（取决于矩阵大小）
- **数值精度**：NumPy精度更高（相对误差~1e-16），NeuMatC精度略低（相对误差~1e-4）
- **内存占用**：NeuMatC需要额外的模型参数存储
- **适用场景**：NeuMatC适合需要多次求解相似矩阵的场景，NumPy适合一次性求解

### 智能算法调度优势
- **自适应选择**：自动选择最适合当前矩阵的求解方法
- **数值稳定**：避免数值不稳定的方法
- **性能优化**：根据矩阵特性选择最快的方法
- **错误处理**：自动处理奇异矩阵等特殊情况

## 参考资料

- [Neural Continuous Mapping for Parametric Matrix Equations](https://example.com)
- PyTorch官方文档：https://pytorch.org/docs/stable/index.html
- NumPy官方文档：https://numpy.org/doc/stable/
- SymPy官方文档：https://docs.sympy.org/
- Streamlit官方文档：https://docs.streamlit.io/

## 致谢

本项目参考了相关领域的研究成果，感谢所有为矩阵计算、深度学习和符号计算领域做出贡献的研究者。特别感谢NeuMatC原始论文的作者，为参数化矩阵计算提供了创新的解决方案。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 更新日志

### v2.0.0 (2026-04-06)
- ✅ 扩展矩阵方程类型支持（AX=B、XA=B、AXB=C等）
- ✅ 增加错误处理和鲁棒性（奇异矩阵检测、秩判断、解类型判断）
- ✅ 优化数值稳定性（使用torch.linalg.solve等稳定方法）
- ✅ 实现智能算法调度（根据矩阵特性自动选择方法）
- ✅ 提高可解释性（输出求解步骤和推理过程）
- ✅ 模块化设计（拆分为parser/、solver/、explainer/、visualizer/）
- ✅ 支持符号计算（集成SymPy）
- ✅ 添加前端交互功能（矩阵变换可视化、步骤展开）

### v1.0.0
- 初始版本，支持矩阵求逆和SVD分解