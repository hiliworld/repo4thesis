import pandas as pd
import matplotlib.pyplot as plt

def plot_smd_data(file_path):
    print(f"正在加载文件: {file_path}")
    
    try:
        # SMD 数据一般是用逗号分隔，且没有表头。如果是空格分隔，请将 sep=',' 改为 sep='\s+'
        df = pd.read_csv(file_path, sep=',', header=None)
        
        print(f"数据加载成功！数据集形状: {df.shape} (行数/时间步, 列数/特征数)")
        
        # 创建画布
        plt.figure(figsize=(16, 8))
        
        # 设定要绘制的特征数量
        # SMD 通常有 38 个特征，全画在一起太乱了，这里默认画前 5 个。你可以修改这个数字。
        num_features_to_plot = min(5, df.shape[1]) 
        
        # 遍历特征并绘制折线图
        for i in range(num_features_to_plot):
            plt.plot(df.iloc[:, i], label=f'Feature {i}', linewidth=1.2, alpha=0.8)
        
        # 设置图表细节
        plt.title('Data Fluctuations in machine-1-1.txt', fontsize=16)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        # 显示图例
        plt.legend(loc='upper right')
        
        # 开启网格线，方便观察波动幅度
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局并显示
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。请检查路径是否准确。")
    except Exception as e:
        print(f"读取或绘图时发生错误: {e}")

if __name__ == "__main__":
    # 替换为你提供的文件路径
    data_file = '/home/sde/MyThesis/data/ServerMachineDataset/train/machine-1-1.txt'
    
    plot_smd_data(data_file)