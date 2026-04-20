import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

from src.data.loader import get_dataloaders
from src.models.anomaly_model import MyFinalModel
from src.utils.loss import ContrastiveLoss
from src.utils.metrics import (
    get_best_f1,
    compute_reference_thresholds,
    apply_dual_threshold_state_machine,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_window_scores(model, x, alpha_pred=1.0, beta_recon=1.0):
    """输出分解后的窗口分数。"""
    pred, recon, _ = model(x)
    pred_score = torch.mean((pred - x[:, -1, :]) ** 2, dim=1)
    recon_last_score = torch.mean((recon[:, -1, :] - x[:, -1, :]) ** 2, dim=1)
    recon_full_score = torch.mean((recon - x) ** 2, dim=(1, 2))
    total_score = alpha_pred * pred_score + beta_recon * recon_last_score
    return {
        "pred_score": pred_score,
        "recon_last_score": recon_last_score,
        "recon_full_score": recon_full_score,
        "total_score": total_score,
        "pred": pred,
        "recon": recon,
    }


def smooth_scores_ema(scores, ema_alpha=0.2):
    if len(scores) == 0:
        return np.array([], dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    out = np.empty_like(scores)
    out[0] = scores[0]
    for i in range(1, len(scores)):
        out[i] = ema_alpha * scores[i] + (1 - ema_alpha) * out[i - 1]
    return out


def collect_reference_scores(model, train_dataset, config):
    """在训练集正常样本上收集参考分数分布。"""
    infer_cfg = config.get('inference', {})
    alpha_pred = float(infer_cfg.get('alpha_pred', 1.0))
    beta_recon = float(infer_cfg.get('beta_recon', 1.0))

    model.eval()
    reference_scores = []
    with torch.no_grad():
        for seq in train_dataset.get_full_sequences():
            if len(seq) < train_dataset.window_size:
                continue
            for t in range(train_dataset.window_size - 1, len(seq)):
                window = seq[t - train_dataset.window_size + 1:t + 1]
                x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
                ws = compute_window_scores(model, x, alpha_pred, beta_recon)
                reference_scores.append(float(ws['total_score'].item()))

    return np.asarray(reference_scores, dtype=np.float32)


def online_rollout_sequence(model, sequence, config):
    """按时间顺序在线滚动打分，支持异常隔离。"""
    infer_cfg = config.get('inference', {})
    window_size = int(config['dataset']['window_size'])
    alpha_pred = float(infer_cfg.get('alpha_pred', 1.0))
    beta_recon = float(infer_cfg.get('beta_recon', 1.0))
    use_isolation = bool(infer_cfg.get('use_isolation', True))
    replacement_mix = float(infer_cfg.get('replacement_mix', 0.5))
    high_threshold = infer_cfg.get('high_threshold', None)
    low_threshold = infer_cfg.get('low_threshold', None)

    seq = np.asarray(sequence, dtype=np.float32)
    sanitized = seq.copy()

    pred_scores, recon_last_scores, recon_full_scores, total_scores = [], [], [], []
    in_anomaly = False

    model.eval()
    with torch.no_grad():
        for t in range(window_size - 1, len(seq)):
            window = sanitized[t - window_size + 1:t + 1].copy()
            window[-1] = seq[t]  # 当前点仍用真实值打分
            x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
            ws = compute_window_scores(model, x, alpha_pred, beta_recon)

            score_total = float(ws['total_score'].item())
            pred_scores.append(float(ws['pred_score'].item()))
            recon_last_scores.append(float(ws['recon_last_score'].item()))
            recon_full_scores.append(float(ws['recon_full_score'].item()))
            total_scores.append(score_total)

            if high_threshold is not None and low_threshold is not None:
                if (not in_anomaly) and score_total >= high_threshold:
                    in_anomaly = True
                elif in_anomaly and score_total <= low_threshold:
                    in_anomaly = False

            if use_isolation and in_anomaly:
                pred_current = ws['pred'][0].detach().cpu().numpy()
                recon_last = ws['recon'][0, -1, :].detach().cpu().numpy()
                replacement = replacement_mix * pred_current + (1.0 - replacement_mix) * recon_last
                sanitized[t] = replacement.astype(np.float32)

    return {
        "pred_score": np.asarray(pred_scores, dtype=np.float32),
        "recon_last_score": np.asarray(recon_last_scores, dtype=np.float32),
        "recon_full_score": np.asarray(recon_full_scores, dtype=np.float32),
        "total_score": np.asarray(total_scores, dtype=np.float32),
    }


def train(args):
    config = load_config(args.config)
    print(f"🔥 Mode: TRAIN | Device: {DEVICE}")
    print(f"📜 Config: {args.config}")

    train_loader, _, input_dim = get_dataloaders(args.config)
    config['dataset']['input_dim'] = input_dim

    model = MyFinalModel(config).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=float(config['train']['lr']))
    criterion_mse = nn.MSELoss()
    criterion_cl = ContrastiveLoss(config['train']['batch_size'], device=DEVICE)

    epochs = config['train']['epochs']
    patience = config['train']['patience']
    best_loss = float('inf')
    patience_counter = 0
    save_path = "best_model.pth"

    print("\n🚀 Start Training...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        start = time.time()

        for batch in train_loader:
            x = batch.to(DEVICE)
            optimizer.zero_grad()

            pred, recon, _ = model(x)

            l_pred = criterion_mse(pred, x[:, -1, :])
            l_recon = criterion_mse(recon, x)

            noise = torch.randn_like(x) * 0.01
            z1 = model.metric_encoder(x)
            z2 = model.metric_encoder(x + noise)
            l_cl = criterion_cl(z1.view(x.size(0), -1), z2.view(x.size(0), -1))

            loss = l_pred + l_recon + 0.1 * l_cl

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        cost = time.time() - start

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Time: {cost:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Saved Best Model ({avg_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early Stopping Triggered.")
                break

    print(f"✅ Training Complete. Model saved to {save_path}")


def evaluate(args):
    config = load_config(args.config)
    print(f"🔥 Mode: TEST | Device: {DEVICE}")

    _, _, input_dim, train_dataset, test_dataset = get_dataloaders(args.config, return_datasets=True)
    config['dataset']['input_dim'] = input_dim

    model = MyFinalModel(config).to(DEVICE)
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file {model_path} not found. Run train first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    infer_cfg = config.get('inference', {})
    ema_alpha = float(infer_cfg.get('ema_alpha', 0.2))
    high_percentile = float(infer_cfg.get('high_percentile', 99.0))
    low_percentile = float(infer_cfg.get('low_percentile', 95.0))

    print("🚀 Collecting reference scores from train(normal) data...")
    reference_scores = collect_reference_scores(model, train_dataset, config)
    high_threshold, low_threshold = compute_reference_thresholds(
        reference_scores,
        high_percentile=high_percentile,
        low_percentile=low_percentile,
    )
    config.setdefault('inference', {})['high_threshold'] = high_threshold
    config['inference']['low_threshold'] = low_threshold
    print(f"📌 Thresholds | high={high_threshold:.6f}, low={low_threshold:.6f}")

    print("🚀 Online rollout over test sequences...")
    total_scores = []
    for seq in tqdm(test_dataset.get_full_sequences()):
        out = online_rollout_sequence(model, seq, config)
        total_scores.append(out['total_score'])

    scores = np.concatenate(total_scores) if total_scores else np.array([], dtype=np.float32)
    scores = smooth_scores_ema(scores, ema_alpha=ema_alpha)

    labels = load_labels(config)
    if labels is None:
        print("⚠️ No labels found. Skipping evaluation metrics.")
        return

    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]

    print("📊 Calculating Metrics...")
    metrics = get_best_f1(labels, scores)
    state_preds = apply_dual_threshold_state_machine(scores, high_threshold, low_threshold)
    dual_f1 = (2 * ((state_preds == 1) & (labels == 1)).sum()) / (
        (state_preds == 1).sum() + (labels == 1).sum() + 1e-10
    )

    print("\n" + "="*40)
    print(f"🌟 FINAL RESULTS ({config['dataset']['name']})")
    print("="*40)
    print(f"AUC            : {metrics['auc']:.4f}")
    print(f"Best F1        : {metrics['best_f1']:.4f}")
    print(f"PA F1          : {metrics['f1_pa']:.4f}")
    print(f"Dual-TH F1     : {dual_f1:.4f}")
    print(f"High/Low TH    : {high_threshold:.6f} / {low_threshold:.6f}")
    print("="*40)


def load_labels(config):
    test_path = config['dataset']['test_file']
    label_path = test_path.replace("test", "test_label")
    pattern = config['dataset']['format']['pattern']
    files = sorted(glob.glob(os.path.join(label_path, pattern)))

    label_list = []
    window = config['dataset']['window_size']

    for f in files:
        try:
            df = pd.read_csv(f, header=None)
            raw = df.values.flatten()
            if len(raw) > window:
                label_list.append(raw[window - 1:])
        except Exception:
            pass

    if not label_list:
        return None
    return np.concatenate(label_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AIOps Anomaly Detection Framework")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Run mode')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        evaluate(args)
