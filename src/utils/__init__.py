import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

def point_adjustment(score, label, thres):
    """AIOps 标准 Point Adjustment"""
    predict = score > thres
    actual = label > 0.5
    anomaly_state = False
    
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]: break
                else:
                    if not predict[j]: predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict

def get_best_f1(labels, scores):
    """计算 Best F1 和 PA-F1"""
    # 1. 计算 AUC
    auc = roc_auc_score(labels, scores)
    
    # 2. 搜索最佳阈值
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_f1 = np.max(f1_scores)
    best_thresh = thresholds[np.argmax(f1_scores)]
    
    # 3. 计算 PA-F1 
    pred_pa = point_adjustment(scores, labels, best_thresh)
    f1_pa = f1_score(labels, pred_pa)
    
    return {
        "auc": auc,
        "best_f1": best_f1,
        "f1_pa": f1_pa,
        "threshold": best_thresh
    }