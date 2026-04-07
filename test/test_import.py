print("开始测试导入")
print("尝试导入 matrix_equation 模块")

try:
    import matrix_equation
    print("导入成功！")
except Exception as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")
