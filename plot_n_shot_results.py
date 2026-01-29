import matplotlib
matplotlib.use('Agg')  # 服务器端防报错
import matplotlib.pyplot as plt

def main():
    # === 1. 填入你实验得到的真实数据 ===
    # X轴: 样本数
    shots = [1, 3, 5]
    
    # 左Y轴: Gap Ratio (倍数)
    # 数据来源: 1-shot(325.0), 3-shot(488.0), 5-shot(354.6)
    gap_ratios = [325.0, 488.0, 354.6]
    
    # 右Y轴: AUC Score
    # 数据来源: 1-shot(0.9960), 3-shot(0.9875), 5-shot(0.9968)
    auc_scores = [0.9960, 0.9875, 0.9968]

    # === 2. 开始画图 ===
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 左轴 (柱状图) ---
    color_bar = '#1f77b4'  # 蓝色
    ax1.set_xlabel('Number of Shots (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gap Ratio (Fault/Normal)', color=color_bar, fontsize=12, fontweight='bold')
    
    # 画柱子
    bars = ax1.bar(shots, gap_ratios, color=color_bar, alpha=0.6, width=0.8, label='Gap Ratio')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.set_ylim(0, 600)  # 根据你的最高值488调整，留点空间
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.1f}x',
                ha='center', va='bottom', color=color_bar, fontweight='bold')

    # --- 右轴 (折线图) ---
    ax2 = ax1.twinx()  
    color_line = '#d62728'  # 红色
    ax2.set_ylabel('AUC Score', color=color_line, fontsize=12, fontweight='bold')
    
    # 画折线
    line = ax2.plot(shots, auc_scores, color=color_line, marker='o', 
                    linewidth=3, markersize=10, label='AUC Score')
    ax2.tick_params(axis='y', labelcolor=color_line)
    
    # 设置AUC的Y轴范围，让波动看起来更明显一点
    # 你的最低是0.9875，最高0.9968，设置在 0.98 到 1.0 之间比较合适
    ax2.set_ylim(0.98, 1.002)

    # 在点上标数值
    for i, txt in enumerate(auc_scores):
        ax2.text(shots[i], txt + 0.001, f'{txt:.4f}', 
                 ha='center', va='bottom', color=color_line, fontweight='bold')

    # --- 装饰 ---
    plt.title('Impact of Shot Number on Model Generalization', fontsize=14, pad=20)
    plt.xticks(shots, [f'{k}-Shot' for k in shots], fontsize=11)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 保存
    save_path = 'n_shot_performance_analysis.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Chart saved to {save_path}")

if __name__ == "__main__":
    main()