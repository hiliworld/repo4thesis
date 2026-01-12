import pandas as pd
import matplotlib.pyplot as plt

# 路径还是要改对
file_path = "/Users/chariesliu/Desktop/MyThesis/ServerMachineDataset/test/machine-1-1.txt"
data = pd.read_csv(file_path, header=None)

# 我们只画第 0 列 (通常是 CPU 利用率) 的前 1000 个时间点
# 这样你看得清楚细节
subset = data.iloc[:1000, 0]

plt.figure(figsize=(12, 4))
plt.plot(subset)
plt.title("Machine 1-1: Feature 0 (First 1000 points)")
plt.xlabel("Time Step")
plt.ylabel("Value (Normalized)")
plt.grid(True)
plt.show()