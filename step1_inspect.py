import pandas as pd
import numpy as np
import os  # <--- 必须导入 os 模块来处理路径
# ==========================================
#这个文件的主要目的就是读取文件 然后打印出读取的文件的一些信息
# ==========================================

# ==========================================
# 核心修改：自动构建绝对路径
# ==========================================
# 1. 获取当前脚本 (step1_inspect.py) 所在的绝对路径
# 例如: /home/sde/MyThesis/
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接数据文件的路径
# 逻辑：当前目录 -> data -> ServerMachineDataset -> test -> machine-1-1.txt
# os.path.join 会自动处理 Linux(/) 和 Windows(\) 的分隔符差异
file_path = os.path.join(current_script_dir, 'data', 'ServerMachineDataset', 'test', 'machine-1-1.txt')

print(f"🔍 [Debug] 正在读取文件路径: {file_path}")

# ==========================================
# 下面是原本的数据读取逻辑
# ==========================================
try:
    # 检查文件是否存在，不存在直接报错提示
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到！请检查路径下是否有该文件: {file_path}")

    # 3. 读取数据
    # SMD 数据通常没有表头 (header=None)
    data = pd.read_csv(file_path, header=None)

    # 4. 打印数据的基本信息
    #主要来说使用的是data.()
    print("\n=== ✅ 数据加载成功！===")
    print(f"数据形状 (行数 Time Steps, 列数 Features): {data.shape}")
    print("提示：SMD 数据集通常有 38 维特征。")

    print("\n=== 前 5 行数据预览 ===")
    print(data.head())

    print("\n=== 数据统计信息 ===")
    print(data.describe())

except FileNotFoundError as e:
    print(f"\n❌ 路径错误: {e}")
    print("建议：请使用 'ls -R data' 命令检查服务器上的文件名是否大小写拼写正确。")
except Exception as e:
    print(f"\n❌ 发生了其他错误: {e}")