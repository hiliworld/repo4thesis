import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from step4_windowing import SMDWindowDataset
from model_v2_with_gat import MyFinalModel

from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

# === 配置 ===
WINDOW_SIZE = 100
BATCH_SIZE = 32
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 1. 加载测试数据和标签 ===
print("正在加载测试数据...")
# 注意：这次我们要读 test 文件夹
test_file_path = "data/ServerMachineDataset/test/machine-1-1.txt"
# 注意：我们要读 test_label 文件夹 (标准答案)
label_file_path = "data/ServerMachineDataset/test_label/machine-1-1.txt"

# 加载数据
dataset = SMDWindowDataset(test_file_path, window_size=WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # 测试集千万不要打乱顺序！

# 加载标签 (Label)
# 标签是一列 0 和 1 (0=正常, 1=异常)
try:
    labels = pd.read_csv(label_file_path, header=None).values
    # 因为我们用了窗口，前99个点没法预测，所以标签也要对应截取
    # 我们预测的是窗口的最后一个点，所以标签从 window_size-1 开始取
    # 但为了对齐方便，我们简单地取后半部分
    # 逻辑：第 i 个窗口预测的是原始数据的第 i + window_size - 1 个点
    labels = labels[WINDOW_SIZE:]
except Exception as e:
    print(f"⚠️ 警告: 没找到标签文件 ({e})，如果是为了演示，我们将假设全是0")
    labels = np.zeros(len(dataset))

# === 2. 加载训练好的模型 ===
model = MyFinalModel(num_features=36).to(DEVICE)
model.load_state_dict(torch.load("my_trained_model.pth", map_location=DEVICE))
model.eval()  # 开启评估模式 (关闭 Dropout)

print("模型加载完毕，开始推理...")

# === 3. 计算异常分数 ===
criterion = nn.MSELoss(reduction='none')  # 不求平均，我们要保留每个样本的 Loss
anomaly_scores = []

with torch.no_grad():  # 测试时不需要算梯度，省内存
    for batch in dataloader:
        x = batch.to(DEVICE)
        target = batch[:, -1, :].to(DEVICE)  # 真实值

        # 预测
        prediction, reconstruction, _ = model(x)

        # 计算 Loss: (预测 - 真实)^2
        # 我们对 36 个指标的 Loss 求平均，得到这一个时间点的总异常分
        loss = criterion(prediction, target)
        loss_per_sample = torch.mean(loss, dim=1)  # [Batch_size]

        # 收集结果
        anomaly_scores.extend(loss_per_sample.cpu().numpy())

# 转成 numpy 方便画图
anomaly_scores = np.array(anomaly_scores)

# 再次对齐标签长度 (防止因为 batch 截断导致长度微小差异)
labels = labels[:len(anomaly_scores)]

print(f"推理完成！生成了 {len(anomaly_scores)} 个异常分数。")

# === 4. 画图 (论文级可视化) ===
plt.figure(figsize=(15, 6))

# A. 画异常分数曲线 (蓝色)
plt.plot(anomaly_scores, label='Anomaly Score (Loss)', color='blue', alpha=0.7, linewidth=1)

# B. 画真实故障区域 (红色背景)
# 只有当 label 为 1 时才画红色
# 这是一个画图小技巧：用 fill_between
time_steps = np.arange(len(labels))
# 将 label 扩展一下以便画图
is_anomaly = labels.flatten() == 1
plt.fill_between(time_steps, 0, np.max(anomaly_scores), where=is_anomaly,
                 color='red', alpha=0.3, label='Ground Truth Anomaly')

plt.title(f"Anomaly Detection Result on Machine 1-1\n(Lower Score = Normal, High Score = Anomaly)", fontsize=14)
plt.xlabel("Time Step (minutes)", fontsize=12)
plt.ylabel("Reconstruction Error (MSE)", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# 保存图片
plt.savefig("result_visualization.png", dpi=300)
print("✅ 图片已保存为 result_visualization.png，请在文件夹中查看！")
plt.show()

# ==========================================
# 请把这段代码复制到 step8 的最后面
# ==========================================

print("\n=== [新增] 开始计算 AUC 和 F1 指标 ===")

try:
    from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

    # 1. 确保数据是 Numpy 格式且长度一致
    # 有时候因为 batch 处理，预测值可能会比标签少几个点，或者反过来
    # 我们取两者的最小长度进行截断，保证一一对应
    min_len = min(len(labels), len(anomaly_scores))
    y_true = labels[:min_len]
    y_scores = anomaly_scores[:min_len]

    # 2. 计算 AUC
    # 如果 y_true 全是 0 (没有异常)，AUC 会报错，所以加个判断
    if np.sum(y_true) > 0:
        auc_score = roc_auc_score(y_true, y_scores)
        print(f"📊 模型 AUC 得分: {auc_score:.4f}")

        # 3. 计算最佳 F1 Score
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_f1 = np.max(f1_scores)
        print(f"🏆 模型最佳 F1 Score: {best_f1:.4f}")

        if auc_score > 0.6:
            print("✅ 评价：模型具备基本的检测能力！(及格线是0.5)")
        if auc_score > 0.8:
            print("🌟 评价：模型表现非常优秀！")

    else:
        print("⚠️ 测试集中没有异常样本 (全是0)，无法计算 AUC/F1。")

except ImportError:
    print("❌ 错误：你还没安装 scikit-learn。请在终端运行: pip install scikit-learn")
except Exception as e:
    print(f"❌ 发生了其他计算错误: {e}")