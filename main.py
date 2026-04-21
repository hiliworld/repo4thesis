import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.data.loader import get_dataloaders
from src.models.anomaly_memory import (
    AnomalyMemoryBank,
    SegmentEmbeddingSchema,
    cluster_anomaly_segments,
)
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


def compute_score_components(outputs, x, alpha_pred=1.0, beta_recon=1.0):
    pred_raw = outputs["pred_raw"]
    pred_corrected = outputs["pred_corrected"]
    recon_raw = outputs["recon_raw"]
    recon_corrected = outputs["recon_corrected"]

    pred_raw_score = torch.mean((pred_raw - x[:, -1, :]) ** 2, dim=1)
    pred_corrected_score = torch.mean((pred_corrected - x[:, -1, :]) ** 2, dim=1)
    recon_raw_last_score = torch.mean((recon_raw[:, -1, :] - x[:, -1, :]) ** 2, dim=1)
    recon_corrected_last_score = torch.mean((recon_corrected[:, -1, :] - x[:, -1, :]) ** 2, dim=1)

    total_raw_score = alpha_pred * pred_raw_score + beta_recon * recon_raw_last_score
    total_corrected_score = alpha_pred * pred_corrected_score + beta_recon * recon_corrected_last_score

    branch_gap = torch.abs(total_corrected_score - total_raw_score)
    correction_norm = torch.mean((outputs["z_corrected"] - outputs["z_fused"]) ** 2, dim=(1, 2))

    node_assign = outputs.get("node_assign")
    if node_assign is not None:
        node_entropy_map = -(node_assign * torch.log(node_assign + 1e-8)).sum(dim=-1)
        node_entropy = node_entropy_map.mean(dim=1)
    else:
        node_entropy = torch.zeros_like(total_raw_score)

    patch_assign = outputs.get("patch_assign")
    if patch_assign is not None:
        patch_entropy_map = -(patch_assign * torch.log(patch_assign + 1e-8)).sum(dim=-1)
        patch_entropy = patch_entropy_map.mean(dim=1)
    else:
        patch_entropy = torch.zeros_like(total_raw_score)

    prototype_uncertainty = 0.5 * (node_entropy + patch_entropy)

    residual_time = torch.mean((recon_corrected - x) ** 2, dim=2)
    last_residual = residual_time[:, -1]
    if residual_time.shape[1] > 1:
        hist_mean = residual_time[:, :-1].mean(dim=1)
    else:
        hist_mean = last_residual
    spike_score = torch.abs(last_residual - hist_mean)

    return {
        "pred_raw_score": pred_raw_score,
        "pred_corrected_score": pred_corrected_score,
        "recon_raw_last_score": recon_raw_last_score,
        "recon_corrected_last_score": recon_corrected_last_score,
        "total_raw_score": total_raw_score,
        "total_corrected_score": total_corrected_score,
        "branch_gap": branch_gap,
        "correction_norm": correction_norm,
        "node_entropy": node_entropy,
        "patch_entropy": patch_entropy,
        "prototype_uncertainty": prototype_uncertainty,
        "spike_score": spike_score,
    }


def _init_reference_buffer():
    return {
        "total_raw_score": [],
        "total_corrected_score": [],
        "branch_gap": [],
        "correction_norm": [],
        "node_entropy": [],
        "patch_entropy": [],
        "prototype_uncertainty": [],
        "spike_score": [],
    }


def collect_reference_statistics(model, train_dataset, config):
    infer_cfg = config.get("inference", {})
    alpha_pred = float(infer_cfg.get("alpha_pred", 1.0))
    beta_recon = float(infer_cfg.get("beta_recon", 1.0))

    model.eval()
    ref_buf = _init_reference_buffer()
    with torch.no_grad():
        for seq in train_dataset.get_full_sequences():
            if len(seq) < train_dataset.window_size:
                continue
            for t in range(train_dataset.window_size - 1, len(seq)):
                window = seq[t - train_dataset.window_size + 1 : t + 1]
                x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
                outputs = model(x)
                comps = compute_score_components(outputs, x, alpha_pred=alpha_pred, beta_recon=beta_recon)
                for k in ref_buf:
                    ref_buf[k].append(float(comps[k].item()))

    stats = {}
    for k, values in ref_buf.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            stats[k] = {"mean": 0.0, "std": 1.0}
        else:
            stats[k] = {"mean": float(arr.mean()), "std": float(arr.std() + 1e-8)}
    return stats


def zscore_with_reference(values, stat_entry):
    return (values - stat_entry["mean"]) / (stat_entry["std"] + 1e-8)


def compute_hybrid_score(components, ref_stats, infer_cfg):
    raw_total_z = zscore_with_reference(components["total_raw_score"], ref_stats["total_raw_score"])
    corrected_total_z = zscore_with_reference(
        components["total_corrected_score"], ref_stats["total_corrected_score"]
    )
    branch_gap_z = zscore_with_reference(components["branch_gap"], ref_stats["branch_gap"])
    correction_norm_z = zscore_with_reference(components["correction_norm"], ref_stats["correction_norm"])
    node_entropy_z = zscore_with_reference(components["node_entropy"], ref_stats["node_entropy"])
    patch_entropy_z = zscore_with_reference(components["patch_entropy"], ref_stats["patch_entropy"])
    prototype_uncertainty_z = zscore_with_reference(
        components["prototype_uncertainty"], ref_stats["prototype_uncertainty"]
    )
    spike_score_z = zscore_with_reference(components["spike_score"], ref_stats["spike_score"])

    w1 = float(infer_cfg.get("hybrid_w_corrected", 1.0))
    w2 = float(infer_cfg.get("hybrid_w_corr_norm", 0.35))
    w3 = float(infer_cfg.get("hybrid_w_branch_gap", 0.25))
    w4 = float(infer_cfg.get("hybrid_w_proto_uncertainty", 0.20))
    w5 = float(infer_cfg.get("hybrid_w_spike", 0.30))

    hybrid_score = (
        w1 * corrected_total_z
        + w2 * correction_norm_z
        + w3 * branch_gap_z
        + w4 * prototype_uncertainty_z
        + w5 * spike_score_z
    )

    return {
        "raw_total_z": raw_total_z,
        "corrected_total_z": corrected_total_z,
        "branch_gap_z": branch_gap_z,
        "correction_norm_z": correction_norm_z,
        "node_entropy_z": node_entropy_z,
        "patch_entropy_z": patch_entropy_z,
        "prototype_uncertainty_z": prototype_uncertainty_z,
        "spike_score_z": spike_score_z,
        "hybrid_score": hybrid_score,
    }


def branch_diagnostics(scores, labels, ref_scores, infer_cfg):
    high_percentile = float(infer_cfg.get("high_percentile", 99.0))
    low_percentile = float(infer_cfg.get("low_percentile", 95.0))
    high, low = compute_reference_thresholds(ref_scores, high_percentile=high_percentile, low_percentile=low_percentile)

    preds = apply_dual_threshold_state_machine(scores, high, low)
    dual_f1 = (2 * ((preds == 1) & (labels == 1)).sum()) / ((preds == 1).sum() + (labels == 1).sum() + 1e-10)

    best = get_best_f1(labels, scores)

    sensitivity = []
    delta_opts = [(-0.5, -0.5), (0.0, 0.0), (0.5, 0.5)]
    for dh, dl in delta_opts:
        h = np.percentile(ref_scores, high_percentile + dh)
        l = np.percentile(ref_scores, low_percentile + dl)
        if l > h:
            l = h
        p = apply_dual_threshold_state_machine(scores, h, l)
        f1 = (2 * ((p == 1) & (labels == 1)).sum()) / ((p == 1).sum() + (labels == 1).sum() + 1e-10)
        sensitivity.append({"high": float(h), "low": float(l), "dual_f1": float(f1)})

    return {
        "metrics": best,
        "dual_f1": float(dual_f1),
        "high": float(high),
        "low": float(low),
        "sensitivity": sensitivity,
    }


def build_anomaly_segments(score_series, high_threshold, low_threshold, min_persistence=1):
    segments = []
    in_anomaly = False
    seg_start = None

    for i, s in enumerate(score_series):
        if (not in_anomaly) and s >= high_threshold:
            in_anomaly = True
            seg_start = i
        elif in_anomaly and s <= low_threshold:
            seg_scores = score_series[seg_start : i + 1]
            length = int(i - seg_start + 1)
            if length >= int(min_persistence):
                segments.append(
                    {
                        "start_idx": int(seg_start),
                        "end_idx": int(i),
                        "length": length,
                        "peak_score": float(np.max(seg_scores)),
                        "mean_score": float(np.mean(seg_scores)),
                    }
                )
            in_anomaly = False
            seg_start = None

    if in_anomaly and seg_start is not None:
        seg_scores = score_series[seg_start:]
        length = int(len(score_series) - seg_start)
        if length >= int(min_persistence):
            segments.append(
                {
                    "start_idx": int(seg_start),
                    "end_idx": int(len(score_series) - 1),
                    "length": length,
                    "peak_score": float(np.max(seg_scores)),
                    "mean_score": float(np.mean(seg_scores)),
                }
            )

    return segments


def online_rollout_sequence(model, sequence, config, ref_stats):
    infer_cfg = config.get("inference", {})
    window_size = int(config["dataset"]["window_size"])
    alpha_pred = float(infer_cfg.get("alpha_pred", 1.0))
    beta_recon = float(infer_cfg.get("beta_recon", 1.0))

    seq = np.asarray(sequence, dtype=np.float32)
    out = {
        "raw_score": [],
        "corrected_score": [],
        "hybrid_score": [],
        "z_fused_mean": [],
        "z_corrected_mean": [],
        "node_assign_hist": [],
        "patch_assign_hist": [],
        "node_delta_mean": [],
        "patch_delta_mean": [],
        "correction_norm": [],
    }

    model.eval()
    with torch.no_grad():
        for t in range(window_size - 1, len(seq)):
            window = seq[t - window_size + 1 : t + 1]
            x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
            outputs = model(x)
            comps = compute_score_components(outputs, x, alpha_pred=alpha_pred, beta_recon=beta_recon)
            zvars = compute_hybrid_score(comps, ref_stats, infer_cfg)

            out["raw_score"].append(float(comps["total_raw_score"].item()))
            out["corrected_score"].append(float(comps["total_corrected_score"].item()))
            out["hybrid_score"].append(float(zvars["hybrid_score"].item()))

            out["z_fused_mean"].append(outputs["z_fused"].mean(dim=1).squeeze(0).detach().cpu().numpy())
            out["z_corrected_mean"].append(outputs["z_corrected"].mean(dim=1).squeeze(0).detach().cpu().numpy())

            node_assign = outputs.get("node_assign")
            if node_assign is not None:
                out["node_assign_hist"].append(node_assign.mean(dim=(0, 1)).detach().cpu().numpy())
            else:
                out["node_assign_hist"].append(None)

            patch_assign = outputs.get("patch_assign")
            if patch_assign is not None:
                out["patch_assign_hist"].append(patch_assign.mean(dim=(0, 1)).detach().cpu().numpy())
            else:
                out["patch_assign_hist"].append(None)

            node_delta = outputs.get("node_delta")
            out["node_delta_mean"].append(float(node_delta.mean().item()) if node_delta is not None else 0.0)

            patch_delta = outputs.get("patch_delta")
            out["patch_delta_mean"].append(float(patch_delta.mean().item()) if patch_delta is not None else 0.0)

            out["correction_norm"].append(float(comps["correction_norm"].item()))

    for k in ["raw_score", "corrected_score", "hybrid_score", "node_delta_mean", "patch_delta_mean", "correction_norm"]:
        out[k] = np.asarray(out[k], dtype=np.float32)
    return out


def export_memory_outputs(config, memory_bank):
    infer_cfg = config.get("inference", {})
    csv_path = infer_cfg.get("anomaly_segment_csv", "outputs/anomaly_segments.csv")
    pkl_path = infer_cfg.get("anomaly_segment_pkl", "outputs/anomaly_segments.pkl")
    cluster_json = infer_cfg.get("anomaly_cluster_json", "outputs/anomaly_clusters.json")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    os.makedirs(os.path.dirname(cluster_json), exist_ok=True)

    memory_bank.export_csv(csv_path)
    memory_bank.save_pkl(pkl_path)

    with open(cluster_json, "w", encoding="utf-8") as f:
        json.dump(memory_bank.cluster_metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved anomaly segments csv: {csv_path}")
    print(f"Saved anomaly segments pkl: {pkl_path}")
    print(f"Saved anomaly clusters json: {cluster_json}")


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
    lambda_node_balance = float(proto_cfg.get("lambda_node_balance", 0.001))
    lambda_patch_balance = float(proto_cfg.get("lambda_patch_balance", 0.001))

    epochs = config["train"]["epochs"]
    patience = config["train"]["patience"]
    best_loss = float("inf")
    patience_counter = 0
    save_path = "best_model.pth"

    print("Start Training")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
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

            l_node_balance = torch.tensor(0.0, device=DEVICE)
            node_assign = outputs.get("node_assign")
            if node_assign is not None:
                mean_node_assign = node_assign.mean(dim=(0, 1))
                uniform_node = torch.full_like(mean_node_assign, 1.0 / mean_node_assign.numel())
                l_node_balance = torch.sum(
                    mean_node_assign * (torch.log(mean_node_assign + 1e-8) - torch.log(uniform_node + 1e-8))
                )

            l_patch_balance = torch.tensor(0.0, device=DEVICE)
            patch_assign = outputs.get("patch_assign")
            if patch_assign is not None:
                mean_patch_assign = patch_assign.mean(dim=(0, 1))
                uniform_patch = torch.full_like(mean_patch_assign, 1.0 / mean_patch_assign.numel())
                l_patch_balance = torch.sum(
                    mean_patch_assign
                    * (torch.log(mean_patch_assign + 1e-8) - torch.log(uniform_patch + 1e-8))
                )

            loss = (
                l_pred
                + l_recon
                + 0.1 * l_cl
                + lambda_node_proto_loss * l_node_proto
                + lambda_patch_proto_loss * l_patch_proto
                + lambda_corr_loss * l_corr
                + lambda_node_balance * l_node_balance
                + lambda_patch_balance * l_patch_balance
            )

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / len(train_loader)
        cost = time.time() - start
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Time: {cost:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model ({avg_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping Triggered")
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
    anomaly_space_cfg = config.get("anomaly_space", {})

    reference_stats = collect_reference_statistics(model, train_dataset, config)

    reference_arrays = {}
    for key, st in reference_stats.items():
        values = []
        for _ in range(2):
            values.append(st["mean"] - st["std"])
            values.append(st["mean"] + st["std"])
        reference_arrays[key] = np.asarray(values, dtype=np.float32)

    raw_ref = []
    corrected_ref = []
    hybrid_ref = []
    with torch.no_grad():
        for seq in train_dataset.get_full_sequences():
            if len(seq) < train_dataset.window_size:
                continue
            for t in range(train_dataset.window_size - 1, len(seq)):
                x = torch.from_numpy(seq[t - train_dataset.window_size + 1 : t + 1]).unsqueeze(0).to(DEVICE)
                outputs = model(x)
                comps = compute_score_components(
                    outputs,
                    x,
                    alpha_pred=float(infer_cfg.get("alpha_pred", 1.0)),
                    beta_recon=float(infer_cfg.get("beta_recon", 1.0)),
                )
                zvars = compute_hybrid_score(comps, reference_stats, infer_cfg)
                raw_ref.append(float(comps["total_raw_score"].item()))
                corrected_ref.append(float(comps["total_corrected_score"].item()))
                hybrid_ref.append(float(zvars["hybrid_score"].item()))

    raw_ref = np.asarray(raw_ref, dtype=np.float32)
    corrected_ref = np.asarray(corrected_ref, dtype=np.float32)
    hybrid_ref = np.asarray(hybrid_ref, dtype=np.float32)

    raw_scores = []
    corrected_scores = []
    hybrid_scores = []

    print("Running Inference")
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(DEVICE)
            outputs = model(x)
            comps = compute_score_components(
                outputs,
                x,
                alpha_pred=float(infer_cfg.get("alpha_pred", 1.0)),
                beta_recon=float(infer_cfg.get("beta_recon", 1.0)),
            )
            zvars = compute_hybrid_score(comps, reference_stats, infer_cfg)

            raw_scores.append(comps["total_raw_score"].detach().cpu().numpy())
            corrected_scores.append(comps["total_corrected_score"].detach().cpu().numpy())
            hybrid_scores.append(zvars["hybrid_score"].detach().cpu().numpy())

    raw_scores = np.concatenate(raw_scores)
    corrected_scores = np.concatenate(corrected_scores)
    hybrid_scores = np.concatenate(hybrid_scores)

    labels = load_labels(config)
    if labels is None:
        print("No labels found. Skipping evaluation metrics.")
        return

    min_len = min(len(raw_scores), len(corrected_scores), len(hybrid_scores), len(labels))
    raw_scores = raw_scores[:min_len]
    corrected_scores = corrected_scores[:min_len]
    hybrid_scores = hybrid_scores[:min_len]
    labels = labels[:min_len]

    raw_diag = branch_diagnostics(raw_scores, labels, raw_ref, infer_cfg)
    corrected_diag = branch_diagnostics(corrected_scores, labels, corrected_ref, infer_cfg)
    hybrid_diag = branch_diagnostics(hybrid_scores, labels, hybrid_ref, infer_cfg)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({config['dataset']['name']})")
    print("=" * 60)
    for name, diag in [
        ("Raw", raw_diag),
        ("Corrected", corrected_diag),
        ("Hybrid", hybrid_diag),
    ]:
        m = diag["metrics"]
        print(f"[{name} Branch]")
        print(f"AUC            : {m['auc']:.4f}")
        print(f"Best F1        : {m['best_f1']:.4f}")
        print(f"PA F1          : {m['f1_pa']:.4f}")
        print(f"Dual-TH F1     : {diag['dual_f1']:.4f}")
        print(f"High threshold : {diag['high']:.6f}")
        print(f"Low threshold  : {diag['low']:.6f}")
        print("Threshold sensitivity:")
        for s in diag["sensitivity"]:
            print(f"  high={s['high']:.6f}, low={s['low']:.6f}, dual_f1={s['dual_f1']:.4f}")
        print("-" * 60)

    if bool(anomaly_space_cfg.get("enable_memory_bank", True)) and bool(infer_cfg.get("save_anomaly_segments", True)):
        memory_bank = AnomalyMemoryBank()
        min_persistence = int(infer_cfg.get("min_anomaly_persistence", 3))

        schema = SegmentEmbeddingSchema(
            latent_dim=int(config.get("model", {}).get("hidden_dim", 64)),
            num_node_prototypes=int(config.get("model", {}).get("prototype", {}).get("num_node_prototypes", 8)),
            num_patch_prototypes=int(config.get("model", {}).get("prototype", {}).get("num_patch_prototypes", 8)),
        )

        corrected_high = corrected_diag["high"]
        corrected_low = corrected_diag["low"]
        hybrid_high = hybrid_diag["high"]
        hybrid_low = hybrid_diag["low"]

        segment_id_counter = 0
        for idx, seq in enumerate(test_dataset.get_full_sequences()):
            file_id = (
                os.path.basename(test_dataset.file_paths[idx])
                if hasattr(test_dataset, "file_paths") and idx < len(test_dataset.file_paths)
                else f"test_seq_{idx}"
            )
            rollout = online_rollout_sequence(model, seq, config, reference_stats)

            branch_to_segments = {
                "corrected": build_anomaly_segments(
                    rollout["corrected_score"], corrected_high, corrected_low, min_persistence=min_persistence
                ),
                "hybrid": build_anomaly_segments(
                    rollout["hybrid_score"], hybrid_high, hybrid_low, min_persistence=min_persistence
                ),
            }

            for branch_name, segments in branch_to_segments.items():
                score_key = f"{branch_name}_score"
                for seg in segments:
                    start_idx = seg["start_idx"]
                    end_idx = seg["end_idx"]
                    node_slice = [x for x in rollout["node_assign_hist"][start_idx : end_idx + 1] if x is not None]
                    patch_slice = [x for x in rollout["patch_assign_hist"][start_idx : end_idx + 1] if x is not None]

                    record = {
                        "segment_id": int(segment_id_counter),
                        "file_id": file_id,
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                        "length": int(seg["length"]),
                        "peak_score": float(np.max(rollout[score_key][start_idx : end_idx + 1])),
                        "mean_score": float(np.mean(rollout[score_key][start_idx : end_idx + 1])),
                        "branch": branch_name,
                        "z_fused_mean": np.mean(
                            np.stack(rollout["z_fused_mean"][start_idx : end_idx + 1], axis=0), axis=0
                        ).tolist(),
                        "z_corrected_mean": np.mean(
                            np.stack(rollout["z_corrected_mean"][start_idx : end_idx + 1], axis=0), axis=0
                        ).tolist(),
                        "node_assign_hist": (
                            np.mean(np.stack(node_slice, axis=0), axis=0).tolist() if node_slice else None
                        ),
                        "patch_assign_hist": (
                            np.mean(np.stack(patch_slice, axis=0), axis=0).tolist() if patch_slice else None
                        ),
                        "node_delta_mean": float(np.mean(rollout["node_delta_mean"][start_idx : end_idx + 1])),
                        "patch_delta_mean": float(np.mean(rollout["patch_delta_mean"][start_idx : end_idx + 1])),
                        "correction_norm_mean": float(np.mean(rollout["correction_norm"][start_idx : end_idx + 1])),
                    }
                    memory_bank.append_segment(record, schema=schema)
                    segment_id_counter += 1

        cluster_anomaly_segments(
            memory_bank,
            method=anomaly_space_cfg.get("clustering_method", "agglomerative"),
            num_clusters=int(anomaly_space_cfg.get("num_anomaly_clusters", 8)),
            unknown_similarity_threshold=float(anomaly_space_cfg.get("unknown_similarity_threshold", 0.55)),
        )
        export_memory_outputs(config, memory_bank)


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
