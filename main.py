import argparse
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.data.loader import get_dataloaders
from src.models.anomaly_model import MyFinalModel
from src.utils.loss import ContrastiveLoss
from src.utils.metrics import (
    apply_dual_threshold_state_machine,
    compute_reference_thresholds,
    get_best_f1,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def compute_window_scores(model, x, alpha_pred=1.0, beta_recon=1.0, branch="corrected"):
    """输出分解后的窗口分数；branch 支持 raw / corrected。"""
    outputs = model(x)
    if branch == "raw":
        pred = outputs["pred_raw"]
        recon = outputs["recon_raw"]
    else:
        pred = outputs["pred_corrected"]
        recon = outputs["recon_corrected"]

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


def collect_reference_scores(model, train_dataset, config, branch="corrected"):
    """在训练集正常样本上收集参考分数分布。"""
    infer_cfg = config.get("inference", {})
    alpha_pred = float(infer_cfg.get("alpha_pred", 1.0))
    beta_recon = float(infer_cfg.get("beta_recon", 1.0))

    model.eval()
    reference_scores = []
    with torch.no_grad():
        for seq in train_dataset.get_full_sequences():
            if len(seq) < train_dataset.window_size:
                continue
            for t in range(train_dataset.window_size - 1, len(seq)):
                window = seq[t - train_dataset.window_size + 1 : t + 1]
                x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
                ws = compute_window_scores(model, x, alpha_pred, beta_recon, branch=branch)
                reference_scores.append(float(ws["total_score"].item()))

    return np.asarray(reference_scores, dtype=np.float32)


def online_rollout_sequence_with_branch(model, sequence, config, branch="corrected"):
    """
    在线滚动打分（分支版）：
    - raw 分支：基于 z_fused
    - corrected 分支：基于 z_corrected
    """
    infer_cfg = config.get("inference", {})
    window_size = int(config["dataset"]["window_size"])
    alpha_pred = float(infer_cfg.get("alpha_pred", 1.0))
    beta_recon = float(infer_cfg.get("beta_recon", 1.0))

    seq = np.asarray(sequence, dtype=np.float32)
    total_scores = []
    z_fused_mean = []
    z_corrected_mean = []
    node_assign_hist = []
    patch_assign_hist = []

    model.eval()
    with torch.no_grad():
        for t in range(window_size - 1, len(seq)):
            window = seq[t - window_size + 1 : t + 1]
            x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
            outputs = model(x)

            if branch == "raw":
                pred = outputs["pred_raw"]
                recon = outputs["recon_raw"]
            else:
                pred = outputs["pred_corrected"]
                recon = outputs["recon_corrected"]

            pred_score = torch.mean((pred - x[:, -1, :]) ** 2, dim=1)
            recon_last = torch.mean((recon[:, -1, :] - x[:, -1, :]) ** 2, dim=1)
            score_total = alpha_pred * pred_score + beta_recon * recon_last
            total_scores.append(float(score_total.item()))

            z_fused_mean.append(float(outputs["z_fused"].mean().item()))
            z_corrected_mean.append(float(outputs["z_corrected"].mean().item()))

            node_assign = outputs.get("node_assign")
            if node_assign is not None:
                node_assign_hist.append(node_assign.mean(dim=(0, 1)).detach().cpu().numpy())
            else:
                node_assign_hist.append(None)

            patch_assign = outputs.get("patch_assign")
            if patch_assign is not None:
                patch_assign_hist.append(patch_assign.mean(dim=(0, 1)).detach().cpu().numpy())
            else:
                patch_assign_hist.append(None)

    return {
        "total_score": np.asarray(total_scores, dtype=np.float32),
        "z_fused_mean": z_fused_mean,
        "z_corrected_mean": z_corrected_mean,
        "node_assign_hist": node_assign_hist,
        "patch_assign_hist": patch_assign_hist,
    }


def build_anomaly_segments(score_series, high_threshold, low_threshold):
    """
    将分数序列转换为连续异常段。
    返回段结构：start_idx/end_idx/length/peak_score/mean_score。
    """
    segments = []
    in_anomaly = False
    seg_start = None

    for i, s in enumerate(score_series):
        if (not in_anomaly) and s >= high_threshold:
            in_anomaly = True
            seg_start = i
        elif in_anomaly and s <= low_threshold:
            seg_scores = score_series[seg_start : i + 1]
            segments.append(
                {
                    "start_idx": int(seg_start),
                    "end_idx": int(i),
                    "length": int(i - seg_start + 1),
                    "peak_score": float(np.max(seg_scores)),
                    "mean_score": float(np.mean(seg_scores)),
                }
            )
            in_anomaly = False
            seg_start = None

    if in_anomaly and seg_start is not None:
        seg_scores = score_series[seg_start:]
        segments.append(
            {
                "start_idx": int(seg_start),
                "end_idx": int(len(score_series) - 1),
                "length": int(len(score_series) - seg_start),
                "peak_score": float(np.max(seg_scores)),
                "mean_score": float(np.mean(seg_scores)),
            }
        )

    return segments


def export_anomaly_segments(config, segment_rows):
    """
    导出异常段：
    - CSV：便于快速查阅
    - PKL：保留复杂字段（如 assignment 直方图）

    字段说明：
    file_id/start_idx/end_idx/length/peak_score/mean_score 是基础异常段信息；
    z_fused_mean/z_corrected_mean/node_assign_hist/patch_assign_hist 是可选原型相关诊断信息。
    """
    infer_cfg = config.get("inference", {})
    csv_path = infer_cfg.get("anomaly_segment_csv", "outputs/anomaly_segments.csv")
    pkl_path = infer_cfg.get("anomaly_segment_pkl", "outputs/anomaly_segments.pkl")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    pd.DataFrame(segment_rows).to_csv(csv_path, index=False)
    with open(pkl_path, "wb") as f:
        pickle.dump(segment_rows, f)

    print(f"Saved anomaly segments csv: {csv_path}")
    print(f"Saved anomaly segments pkl: {pkl_path}")


def train(args):
    config = load_config(args.config)
    print(f"Mode: TRAIN | Device: {DEVICE}")
    print(f"Config: {args.config}")

    train_loader, _, input_dim = get_dataloaders(args.config)
    config["dataset"]["input_dim"] = input_dim

    model = MyFinalModel(config).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    criterion_mse = nn.MSELoss()
    criterion_cl = ContrastiveLoss(config["train"]["batch_size"], device=DEVICE)

    proto_cfg = config.get("model", {}).get("prototype", {})
    lambda_node_proto_loss = float(proto_cfg.get("lambda_node_proto_loss", 0.02))
    lambda_patch_proto_loss = float(proto_cfg.get("lambda_patch_proto_loss", 0.02))
    lambda_corr_loss = float(proto_cfg.get("lambda_corr_loss", 0.005))

    epochs = config["train"]["epochs"]
    patience = config["train"]["patience"]
    best_loss = float("inf")
    patience_counter = 0
    save_path = "best_model.pth"

    print("\nStart Training...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        start = time.time()

        for batch in train_loader:
            x = batch.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(x)
            pred = outputs["pred"]
            recon = outputs["recon"]

            l_pred = criterion_mse(pred, x[:, -1, :])
            l_recon = criterion_mse(recon, x)

            noise = torch.randn_like(x) * 0.01
            z1 = model.metric_encoder(x)
            z2 = model.metric_encoder(x + noise)
            l_cl = criterion_cl(z1.view(x.size(0), -1), z2.view(x.size(0), -1))

            l_node_proto = torch.tensor(0.0, device=DEVICE)
            node_proto_latent = outputs.get("node_proto_latent")
            z_fused = outputs.get("z_fused")
            if node_proto_latent is not None and z_fused is not None:
                l_node_proto = torch.mean((node_proto_latent - z_fused) ** 2)

            l_patch_proto = torch.tensor(0.0, device=DEVICE)
            patch_proto_latent = outputs.get("patch_proto_latent")
            z_patch = outputs.get("z_patch")
            if patch_proto_latent is not None and z_patch is not None:
                l_patch_proto = torch.mean((patch_proto_latent - z_patch) ** 2)

            l_corr = torch.tensor(0.0, device=DEVICE)
            z_corrected = outputs.get("z_corrected")
            if z_corrected is not None and z_fused is not None:
                l_corr = torch.mean((z_corrected - z_fused) ** 2)

            loss = (
                l_pred
                + l_recon
                + 0.1 * l_cl
                + lambda_node_proto_loss * l_node_proto
                + lambda_patch_proto_loss * l_patch_proto
                + lambda_corr_loss * l_corr
            )

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
            print(f"   Saved Best Model ({avg_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping Triggered.")
                break

    print(f"Training Complete. Model saved to {save_path}")


def evaluate(args):
    config = load_config(args.config)
    print(f"Mode: TEST | Device: {DEVICE}")

    _, test_loader, input_dim, train_dataset, test_dataset = get_dataloaders(args.config, return_datasets=True)
    config["dataset"]["input_dim"] = input_dim

    model = MyFinalModel(config).to(DEVICE)
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Run train first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    infer_cfg = config.get("inference", {})

    corrected_ref_scores = collect_reference_scores(model, train_dataset, config, branch="corrected")
    raw_ref_scores = collect_reference_scores(model, train_dataset, config, branch="raw")

    corrected_high, corrected_low = compute_reference_thresholds(
        corrected_ref_scores,
        high_percentile=float(infer_cfg.get("high_percentile", 99.0)),
        low_percentile=float(infer_cfg.get("low_percentile", 95.0)),
    )
    raw_high, raw_low = compute_reference_thresholds(
        raw_ref_scores,
        high_percentile=float(infer_cfg.get("high_percentile", 99.0)),
        low_percentile=float(infer_cfg.get("low_percentile", 95.0)),
    )

    raw_scores = []
    corrected_scores = []
    node_assign_acc = []
    patch_assign_acc = []

    print("Running Inference...")
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(DEVICE)
            outputs = model(x)

            pred_raw = outputs["pred_raw"]
            recon_raw = outputs["recon_raw"]
            pred_corr = outputs["pred_corrected"]
            recon_corr = outputs["recon_corrected"]

            l_pred_raw = torch.mean((pred_raw - x[:, -1, :]) ** 2, dim=1)
            l_recon_raw = torch.mean((recon_raw - x) ** 2, dim=(1, 2))
            l_pred_corr = torch.mean((pred_corr - x[:, -1, :]) ** 2, dim=1)
            l_recon_corr = torch.mean((recon_corr - x) ** 2, dim=(1, 2))

            raw_scores.append((l_pred_raw + l_recon_raw).cpu().numpy())
            corrected_scores.append((l_pred_corr + l_recon_corr).cpu().numpy())

            if outputs.get("node_assign") is not None:
                node_assign_acc.append(outputs["node_assign"].mean(dim=(0, 1)).detach().cpu().numpy())
            if outputs.get("patch_assign") is not None:
                patch_assign_acc.append(outputs["patch_assign"].mean(dim=(0, 1)).detach().cpu().numpy())

    raw_scores = np.concatenate(raw_scores)
    corrected_scores = np.concatenate(corrected_scores)

    labels = load_labels(config)
    if labels is None:
        print("No labels found. Skipping evaluation metrics.")
        return

    min_len = min(len(raw_scores), len(corrected_scores), len(labels))
    raw_scores = raw_scores[:min_len]
    corrected_scores = corrected_scores[:min_len]
    labels = labels[:min_len]

    print("Calculating Metrics...")
    raw_metrics = get_best_f1(labels, raw_scores)
    corrected_metrics = get_best_f1(labels, corrected_scores)

    raw_state_preds = apply_dual_threshold_state_machine(raw_scores, raw_high, raw_low)
    corrected_state_preds = apply_dual_threshold_state_machine(corrected_scores, corrected_high, corrected_low)

    raw_dual_f1 = (2 * ((raw_state_preds == 1) & (labels == 1)).sum()) / (
        (raw_state_preds == 1).sum() + (labels == 1).sum() + 1e-10
    )
    corrected_dual_f1 = (2 * ((corrected_state_preds == 1) & (labels == 1)).sum()) / (
        (corrected_state_preds == 1).sum() + (labels == 1).sum() + 1e-10
    )

    print("\n" + "=" * 40)
    print(f"FINAL RESULTS ({config['dataset']['name']})")
    print("=" * 40)
    print("[Raw Branch]")
    print(f"AUC            : {raw_metrics['auc']:.4f}")
    print(f"Best F1        : {raw_metrics['best_f1']:.4f}")
    print(f"PA F1          : {raw_metrics['f1_pa']:.4f}")
    print(f"Dual-TH F1     : {raw_dual_f1:.4f}")
    print(f"High/Low TH    : {raw_high:.6f} / {raw_low:.6f}")
    print("-" * 40)
    print("[Corrected Branch]")
    print(f"AUC            : {corrected_metrics['auc']:.4f}")
    print(f"Best F1        : {corrected_metrics['best_f1']:.4f}")
    print(f"PA F1          : {corrected_metrics['f1_pa']:.4f}")
    print(f"Dual-TH F1     : {corrected_dual_f1:.4f}")
    print(f"High/Low TH    : {corrected_high:.6f} / {corrected_low:.6f}")
    print("=" * 40)

    if node_assign_acc:
        node_usage = np.mean(np.stack(node_assign_acc, axis=0), axis=0)
        node_entropy = -np.sum(node_usage * np.log(node_usage + 1e-10))
        print("Node prototype usage:", np.array2string(node_usage, precision=4))
        print(f"Node assignment entropy: {node_entropy:.6f}")
    else:
        print("Node prototype usage: unavailable")

    if patch_assign_acc:
        patch_usage = np.mean(np.stack(patch_assign_acc, axis=0), axis=0)
        patch_entropy = -np.sum(patch_usage * np.log(patch_usage + 1e-10))
        print("Patch prototype usage:", np.array2string(patch_usage, precision=4))
        print(f"Patch assignment entropy: {patch_entropy:.6f}")
    else:
        print("Patch prototype usage: unavailable")

    if bool(infer_cfg.get("save_anomaly_segments", True)):
        all_segments = []
        for idx, seq in enumerate(test_dataset.get_full_sequences()):
            file_id = (
                os.path.basename(test_dataset.file_paths[idx])
                if hasattr(test_dataset, "file_paths") and idx < len(test_dataset.file_paths)
                else f"test_seq_{idx}"
            )
            rollout = online_rollout_sequence_with_branch(model, seq, config, branch="corrected")
            segs = build_anomaly_segments(rollout["total_score"], corrected_high, corrected_low)

            for seg in segs:
                start_idx = seg["start_idx"]
                end_idx = seg["end_idx"]
                node_slice = [x for x in rollout["node_assign_hist"][start_idx : end_idx + 1] if x is not None]
                patch_slice = [x for x in rollout["patch_assign_hist"][start_idx : end_idx + 1] if x is not None]

                seg["file_id"] = file_id
                seg["z_fused_mean"] = float(np.mean(rollout["z_fused_mean"][start_idx : end_idx + 1]))
                seg["z_corrected_mean"] = float(np.mean(rollout["z_corrected_mean"][start_idx : end_idx + 1]))
                seg["node_assign_hist"] = (
                    np.mean(np.stack(node_slice, axis=0), axis=0).tolist() if node_slice else None
                )
                seg["patch_assign_hist"] = (
                    np.mean(np.stack(patch_slice, axis=0), axis=0).tolist() if patch_slice else None
                )
                all_segments.append(seg)

        export_anomaly_segments(config, all_segments)


def load_labels(config):
    test_path = config["dataset"]["test_file"]
    label_path = test_path.replace("test", "test_label")
    pattern = config["dataset"]["format"]["pattern"]
    files = sorted(glob.glob(os.path.join(label_path, pattern)))

    label_list = []
    window = config["dataset"]["window_size"]

    for f in files:
        try:
            df = pd.read_csv(f, header=None)
            raw = df.values.flatten()
            if len(raw) > window:
                label_list.append(raw[window - 1 :])
        except Exception:
            pass

    if not label_list:
        return None
    return np.concatenate(label_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIOps Anomaly Detection Framework")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Run mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        evaluate(args)
