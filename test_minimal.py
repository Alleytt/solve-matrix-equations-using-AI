print("测试 1: 简单打印")
print("Hello World!")

print("\n测试 2: 导入torch")
import torch
print("导入成功")
print(f"Torch版本: {torch.__version__}")

print("\n测试 3: 创建张量")
t = torch.tensor([1, 2, 3])
print(f"张量: {t}")

print("\n测试 4: 矩阵乘法")
a = torch.eye(2)
b = torch.ones(2, 2)
c = a @ b
print(f"结果: {c}")

print("\n测试 5: 定义简单模型")
import torch.nn as nn
class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()
print("模型创建成功")

print("\n测试 6: 前向传播")
x = torch.tensor([1, 2, 3])
y = model(x)
print(f"输出: {y}")

print("\n测试 7: 损失计算")
target = torch.tensor([2, 4, 6])
l = nn.MSELoss()(y, target)
print(f"损失: {l}")

print("\n测试 8: 反向传播")
l.backward()
print("反向传播成功")

print("\n所有测试完成")
