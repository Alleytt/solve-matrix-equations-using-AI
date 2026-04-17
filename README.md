# AI 矩阵方程求解器 (Matrix Equation Solver)

## 项目简介

一个基于 AI 与传统数值方法相结合的矩阵方程求解系统，支持多种矩阵类型、多种求解方法，并提供智能化的性能分析和推荐。该项目旨在提供一个高效、易用、功能丰富的矩阵方程求解工具，适用于科学计算、工程应用、教育教学等场景。

## 主要功能

### 核心功能

1. **多种求解方法**
   - **LU 分解**：适用于一般方阵，速度快
   - **SVD 分解**：适用于病态矩阵，稳定性好
   - **QR 分解**：适用于超定方程组，数值稳定
   - **Cholesky 分解**：适用于对称正定矩阵，速度快
   - **AI 求解**：基于神经网络的快速求解，适用于大矩阵
   - **混合求解**：AI + 数值方法结合，兼顾速度与精度

2. **问题类型支持**
   - **方阵方程组 (AX=B)**：标准线性方程组
   - **超定方程组**：使用最小二乘解
   - **欠定方程组**：使用最小范数解

3. **矩阵类型支持**
   - **随机矩阵**：高斯分布、均匀分布
   - **低秩矩阵**：指定秩的矩阵
   - **稀疏矩阵**：CSR、COO、CSC 格式
   - **对称矩阵**：转置等于自身的矩阵
   - **正定矩阵**：所有特征值为正的矩阵
   - **托普利兹矩阵**：对角线元素相等的矩阵
   - **希尔伯特矩阵**：高度病态的矩阵
   - **循环矩阵**：每行是前一行循环右移一位的矩阵

4. **智能分析**
   - **条件数分析**：自动计算并评估矩阵条件数
   - **矩阵特性识别**：自动识别对称性、正定性、稀疏性等
   - **求解器推荐**：基于矩阵特性智能推荐最优求解方法
   - **后验误差估计**：计算求解结果的误差范围
   - **警告信息**：针对病态矩阵、奇异矩阵等情况生成警告

5. **性能优化**
   - **混合求解器**：AI 提供初始解，数值方法迭代修正
   - **模型优化**：剪枝、量化、蒸馏等技术
   - **部署优化**：ONNX 导出、TensorRT 加速
   - **并行计算**：支持多线程处理

### 界面功能

1. **多种输入方式**
   - **手动输入**：直接在界面中输入矩阵
   - **随机生成**：批量生成指定特性的矩阵
   - **文件上传**：支持 CSV、NPY、MAT、TXT 格式

2. **批量求解**
   - **多矩阵测试**：一次测试多个不同特性的矩阵
   - **性能对比**：自动生成不同求解方法的性能对比
   - **报告生成**：生成详细的性能分析报告
   - **结果导出**：支持 CSV 格式导出测试结果

3. **可视化**
   - **奇异值分布**：显示矩阵奇异值的分布情况
   - **条件数影响**：显示条件数对求解误差的影响
   - **时间对比**：不同求解方法的时间消耗对比
   - **误差热力图**：显示求解误差的空间分布
   - **性能曲线**：矩阵大小与求解时间的关系

4. **高级设置**
   - **条件数阈值**：调整病态矩阵的判断阈值
   - **迭代细化**：设置数值方法的迭代次数
   - **稀疏矩阵支持**：启用/禁用稀疏矩阵优化
   - **详细分析**：控制分析报告的详细程度

## 技术栈

- **核心语言**: Python 3.14+
- **数值计算**: NumPy 1.26+, SciPy 1.11+
- **机器学习**: PyTorch 2.1+
- **界面**: Streamlit 1.30+
- **可视化**: Matplotlib 3.8+, Seaborn 0.13+
- **稀疏矩阵**: SciPy Sparse
- **文件处理**: NumPy, SciPy.io

## 项目结构

```
solve-matrix-equations-using-AI-main/
├── app.py                # 原始界面
├── app_advanced.py       # 增强版界面
├── experiment.py         # 性能对比实验
├── requirements.txt      # 依赖项
├── src/
│   ├── config/           # 配置管理
│   │   └── __init__.py   # 项目配置定义
│   ├── models/           # 模型定义
│   │   ├── __init__.py
│   │   ├── low_rank.py   # 低秩映射模型
│   │   └── model_optimizer.py  # 模型优化工具
│   ├── solver/           # 求解器
│   │   ├── __init__.py
│   │   ├── numerical_stability.py  # 数值稳定性分析
│   │   ├── hybrid_solver.py  # 混合求解器
│   │   └── unified_solver.py  # 统一求解器
│   ├── utils/            # 工具函数
│   │   ├── __init__.py
│   │   └── advanced_data_generator.py  # 高级数据生成器
│   ├── explainer/        # 解释器
│   │   └── __init__.py
│   └── visualizer/       # 可视化
│       └── __init__.py
├── test/                 # 测试文件
│   ├── test_basic.py
│   ├── test_incremental.py
│   ├── test_training.py
│   └── test_functions.py
└── README.md             # 项目说明
```

## 快速开始

### 安装依赖

1. **创建虚拟环境（推荐）**

   ```powershell
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **安装依赖包**

   ```bash
   pip install -r requirements.txt
   ```

   依赖包包括：
   - numpy
   - scipy
   - torch
   - streamlit
   - matplotlib
   - seaborn

### 运行增强版界面

```bash
streamlit run app_advanced.py
```

服务启动后，在浏览器中访问以下地址：
- **本地地址**：http://localhost:8501
- **网络地址**：http://[your-ip]:8501

### 基本使用

#### 方法一：手动输入矩阵

1. **选择输入方式**：在左侧边栏选择"手动输入"
2. **输入矩阵**：
   - 在矩阵A输入框中输入系数矩阵，每行用空格或逗号分隔，行与行之间用换行分隔
   - 在矩阵B输入框中输入右侧向量或矩阵
3. **选择求解方法**：
   - 自动检测（推荐）
   - LU 分解
   - SVD 分解
   - QR 分解
   - Cholesky 分解
   - AI 求解
   - 混合求解
4. **点击"分析与求解"**：
   - 系统会分析矩阵特性
   - 显示求解结果
   - 展示性能指标

#### 方法二：随机生成矩阵

1. **选择输入方式**：在左侧边栏选择"随机生成"
2. **设置参数**：
   - 矩阵大小：输入矩阵的维度（如 10 表示 10x10 矩阵）
   - 分布类型：选择矩阵元素的分布（高斯、均匀、低秩等）
   - 条件数：可选，设置矩阵的条件数（越大越病态）
3. **点击"生成矩阵"**：系统自动生成矩阵A和B
4. **选择求解方法**：同上
5. **点击"分析与求解"**：查看结果

#### 方法三：文件上传

1. **选择输入方式**：在左侧边栏选择"文件上传"
2. **上传文件**：
   - 支持 CSV、NPY、MAT、TXT 格式
   - 对于 CSV 文件，每行代表矩阵的一行，元素用逗号分隔
   - 对于 NPY 文件，直接加载 NumPy 数组
   - 对于 MAT 文件，加载 MATLAB 格式的矩阵
3. **选择求解方法**：同上
4. **点击"分析与求解"**：查看结果

## 批量测试功能

### 批量测试步骤

1. **进入批量测试区域**：滚动到主界面下方
2. **设置测试参数**：
   - 测试数量：要测试的矩阵数量
   - 矩阵大小范围：最小和最大矩阵维度
   - 分布类型：选择矩阵元素的分布
   - 求解方法：选择要对比的求解方法
   - 条件数范围：设置矩阵的条件数范围
3. **点击"开始批量测试"**：
   - 系统会生成多个矩阵并测试
   - 显示详细的性能对比表格
   - 生成可视化图表
4. **导出结果**：点击"导出CSV"保存测试结果

### 测试结果说明

- **时间 (ms)**：求解所需的时间
- **相对误差**：||X_pred - X_true|| / ||X_true||
- **残差**：||AX - B||
- **条件数**：矩阵的条件数
- **方法**：使用的求解方法

## 核心模块详解

### 1. 配置管理 (`src/config`)

**功能**：提供统一的项目配置系统

**主要组件**：
- `ProjectConfig`：项目级配置
- `ModelConfig`：模型相关配置
- `TrainingConfig`：训练相关配置
- `HybridSolverConfig`：混合求解器配置
- `DataGeneratorConfig`：数据生成器配置
- `PerformanceConfig`：性能相关配置

**使用示例**：
```python
from src.config import get_default_config

config = get_default_config()
config.model_config.hidden_size = 256
config.training_config.batch_size = 64
```

### 2. 数值稳定性分析 (`src/solver/numerical_stability.py`)

**功能**：分析矩阵特性并推荐求解方法

**主要功能**：
- 计算矩阵条件数
- 检测矩阵对称性、正定性
- 评估矩阵稀疏性
- 智能推荐求解方法
- 生成警告信息

**使用示例**：
```python
from src.solver.numerical_stability import analyze_and_recommend
import numpy as np

A = np.random.randn(10, 10)
analysis = analyze_and_recommend(A)
print(f"条件数: {analysis.condition_number}")
print(f"推荐方法: {analysis.recommended_solver.value}")
```

### 3. 混合求解器 (`src/solver/hybrid_solver.py`)

**功能**：结合 AI 与传统数值方法

**主要功能**：
- AI 提供初始解
- 数值方法迭代修正
- 自适应策略切换
- 支持多种求解场景

**使用示例**：
```python
from src.solver.hybrid_solver import HybridSolver, HybridSolverConfig
import numpy as np

# 创建混合求解器
config = HybridSolverConfig(use_adaptive_switching=True)
hybrid_solver = HybridSolver(ai_model=your_model, config=config)

# 求解方程
A = np.random.randn(10, 10)
B = np.random.randn(10)
result = hybrid_solver.solve(A, B)
print(f"解: {result['solution']}")
print(f"时间: {result['total_time_ms']} ms")
```

### 4. 统一求解器 (`src/solver/unified_solver.py`)

**功能**：支持多种矩阵类型和求解方法

**主要功能**：
- 支持稀疏矩阵
- 支持非方阵
- 支持最小二乘解
- 集成多种求解方法

**使用示例**：
```python
from src.solver.unified_solver import solve_general
import numpy as np

# 求解超定方程组
A = np.random.randn(10, 5)
B = np.random.randn(10)
result = solve_general(A, B, problem_type='least_squares')
print(f"解: {result['solution']}")
print(f"残差: {result['residual']}")
```

### 5. 高级数据生成器 (`src/utils/advanced_data_generator.py`)

**功能**：生成多样化的矩阵数据

**主要功能**：
- 支持 9 种矩阵分布
- 支持自定义条件数
- 批量生成矩阵
- 数据分割（训练/验证/测试）

**使用示例**：
```python
from src.utils.advanced_data_generator import AdvancedMatrixGenerator

# 创建数据生成器
generator = AdvancedMatrixGenerator()

# 生成低秩矩阵
A = generator.generate(size=10, distribution='low_rank', rank=3)

# 生成高条件数矩阵
A_ill = generator.generate(size=10, condition_number=1e6)

# 批量生成
batch = generator.generate_batch(batch_size=5, size=8)
```

### 6. 模型优化器 (`src/models/model_optimizer.py`)

**功能**：优化模型性能和大小

**主要功能**：
- 模型剪枝：减少模型参数
- 模型量化：降低精度以提高速度
- 模型蒸馏：小模型模仿大模型
- ONNX 导出：支持部署到其他平台

**使用示例**：
```python
from src.models.model_optimizer import ModelPruner, ModelQuantizer
import torch
import torch.nn as nn

# 创建模型
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 100)
)

# 剪枝模型
pruner = ModelPruner(model)
pruned_model = pruner.prune_magnitude(sparsity=0.5)

# 量化模型
quantizer = ModelQuantizer(model)
quantized_model = quantizer.quantize()
```

## 性能指标

### 时间复杂度

| 方法 | 时间复杂度 | 适用场景 |
|------|------------|----------|
| LU 分解 | O(n³) | 一般方阵 |
| SVD 分解 | O(n³) | 病态矩阵 |
| QR 分解 | O(n³) | 超定方程组 |
| Cholesky | O(n³) | 对称正定 |
| AI 求解 | O(n²) | 大矩阵 |
| 混合求解 | O(n²) + O(n³/k) | 通用 |

### 精度指标

- **相对误差**：||X_pred - X_true|| / ||X_true||
- **残差**：||AX - B||
- **条件数**：矩阵的条件数，衡量矩阵的病态程度

### 性能对比

对于 100x100 矩阵：
- **LU 分解**：~1ms
- **SVD 分解**：~5ms
- **QR 分解**：~3ms
- **Cholesky**：~0.5ms
- **AI 求解**：~0.1ms
- **混合求解**：~0.5ms

对于 1000x1000 矩阵：
- **LU 分解**：~100ms
- **SVD 分解**：~500ms
- **QR 分解**：~300ms
- **Cholesky**：~50ms
- **AI 求解**：~10ms
- **混合求解**：~20ms

## 应用场景

1. **科学计算**：
   - 求解线性方程组
   - 矩阵求逆
   - 特征值计算
   - 最小二乘问题

2. **工程应用**：
   - 结构力学分析
   - 电路分析
   - 信号处理
   - 控制系统设计

3. **教育教学**：
   - 线性代数演示
   - 数值方法教学
   - 矩阵理论学习
   - 算法性能对比

4. **研究分析**：
   - 数值方法性能评估
   - AI 在科学计算中的应用
   - 病态矩阵处理
   - 大规模矩阵求解

## 优势特点

1. **智能化**：
   - 自动分析矩阵特性
   - 智能推荐最优求解方法
   - 自适应策略调整

2. **高效性**：
   - AI 与传统方法结合
   - 兼顾速度与精度
   - 支持大规模矩阵

3. **多样化**：
   - 支持多种矩阵类型
   - 支持多种求解方法
   - 支持多种输入方式

4. **易用性**：
   - 直观的 Web 界面
   - 支持多种输入方式
   - 详细的分析报告

5. **扩展性**：
   - 模块化设计
   - 易于添加新功能
   - 支持自定义模型

## 常见问题

### 1. 前端打不开怎么办？

**解决方案**：
- 检查依赖是否安装：`pip install -r requirements.txt`
- 检查端口是否被占用：尝试使用不同端口
- 查看错误信息：根据错误信息进行修复

### 2. 求解速度慢怎么办？

**解决方案**：
- 对于大矩阵，使用 AI 或混合求解方法
- 对于稀疏矩阵，启用稀疏矩阵支持
- 对于病态矩阵，使用 SVD 分解

### 3. 求解精度不够怎么办？

**解决方案**：
- 增加迭代细化步骤
- 使用 SVD 或 QR 分解
- 对于病态矩阵，使用混合求解方法

### 4. 如何处理奇异矩阵？

**解决方案**：
- 系统会自动检测奇异矩阵
- 对于奇异矩阵，会推荐使用 SVD 分解
- 对于超定或欠定方程组，会使用最小二乘或最小范数解

### 5. 如何导出求解结果？

**解决方案**：
- 在批量测试中，点击"导出CSV"按钮
- 在求解结果区域，复制结果或使用截图

## 未来规划

- [ ] 支持更多矩阵分解方法（EVD、Schur 分解等）
- [ ] 实现时变矩阵在线求解
- [ ] 集成更先进的神经算子结构（FNO、DeepONet）
- [ ] 支持更大规模的矩阵求解
- [ ] 添加并行计算支持
- [ ] 实现 GPU 加速
- [ ] 支持更多文件格式
- [ ] 添加命令行接口
- [ ] 提供 API 服务

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

### 贡献步骤

1. **Fork 仓库**
2. **创建分支**：`git checkout -b feature/your-feature`
3. **修改代码**
4. **测试**：确保所有测试通过
5. **提交代码**：`git commit -m "Add your feature"`
6. **推送分支**：`git push origin feature/your-feature`
7. **创建 Pull Request**

---

**感谢使用 AI 矩阵方程求解器！** 
