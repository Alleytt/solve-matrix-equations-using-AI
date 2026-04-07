print("开始测试模块导入")

print("\n1. 导入torch")
import torch
print("导入成功")

print("\n2. 导入torch.nn")
import torch.nn as nn
print("导入成功")

print("\n3. 导入torch.optim")
import torch.optim as optim
print("导入成功")

print("\n4. 导入numpy")
import numpy as np
print("导入成功")

print("\n5. 导入time")
import time
print("导入成功")

print("\n6. 导入matplotlib.pyplot")
try:
    import matplotlib.pyplot as plt
    print("导入成功")
except Exception as e:
    print(f"导入失败: {e}")

print("\n7. 导入tqdm")
try:
    from tqdm import tqdm
    print("导入成功")
except Exception as e:
    print(f"导入失败: {e}")

print("\n所有模块导入测试完成")
