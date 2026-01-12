import pandas as pd
import numpy as np

# 1. 设置文件路径 (请修改为你自己的路径！)
# 比如 Windows 可能是: r"D:\Project\ServerMachineDataset\train\machine-1-1.txt"
file_path = "/Users/chariesliu/Desktop/MyThesis/ServerMachineDataset/test/machine-1-1.txt"

try:
    # 2. 读取数据
    # SMD 通常是逗号分隔的，没有表头(header=None)
    data = pd.read_csv(file_path, header=None)

    # 3. 打印数据的基本信息
    print("=== 数据加载成功！===")
    print(f"数据形状 (行数, 列数): {data.shape}")
    print("行数 (Time Steps) 代表时间点数量")
    print("列数 (Features) 代表指标数量 (SMD应该是38)")

    print("\n=== 前 5 行数据长这样 ===")
    print(data.head())

    # 4. 看看数据是不是只有数字
    print("\n=== 数据统计信息 ===")
    print(data.describe())

except FileNotFoundError:
    print("错误：找不到文件！请检查 file_path 这一行是否写对了路径。")
except Exception as e:
    print(f"发生了其他错误: {e}")