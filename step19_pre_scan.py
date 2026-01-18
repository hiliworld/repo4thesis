import os
import glob
import pandas as pd
from tqdm import tqdm

METRIC_DIR = "data/processed_metrics"

def find_optimal_dimension():
    print("🔍 正在扫描所有指标文件，寻找最大特征维度...")
    files = glob.glob(os.path.join(METRIC_DIR, "*.csv"))
    
    max_dim = 0
    max_file = ""
    dims = []
    
    for f in tqdm(files):
        try:
            # 只读一行，速度极快
            df = pd.read_csv(f, nrows=1)
            # 减去 timestamp 这一列
            curr_dim = len(df.columns) - 1
            dims.append(curr_dim)
            
            if curr_dim > max_dim:
                max_dim = curr_dim
                max_file = os.path.basename(f)
        except:
            pass
            
    print("\n" + "="*40)
    print(f"📊 维度统计报告")
    print("="*40)
    print(f"最大维度: {max_dim} (出现在 {max_file})")
    print(f"最小维度: {min(dims)}")
    print(f"平均维度: {sum(dims)/len(dims):.2f}")
    print(f"建议 FIXED_FEATURE_DIM 设置为: {max_dim}")
    print("="*40)

if __name__ == "__main__":
    find_optimal_dimension()