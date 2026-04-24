import argparse
import glob
import json
import os
import shutil
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
    build_learnable_segment_repr,
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


def attach_window_repr_from_components(model, outputs, components):
    """Attach score-aware window representation if learnable segment encoder is enabled."""
    if getattr(model, "segment_encoder", None) is None:
        return outputs
    score_stats = torch.stack(
        [
            components["total_corrected_score"],
            components["correction_norm"],
            components["branch_gap"],
            components["spike_score"],
        ],
        dim=-1,
    )  # [B, 4]
    outputs["score_stats"] = score_stats
    outputs["window_repr"] = model.segment_encoder.encode_window(
        z_local_slots=outputs["z_local_slots"],
        z_global_slots=outputs["z_global_slots"],
        z_corrected=outputs["z_corrected"],
        score_stats=score_stats,
    )
    return outputs


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
        "window_repr_seq": [],
    }

    model.eval()
    with torch.no_grad():
        for t in range(window_size - 1, len(seq)):
            window = seq[t - window_size + 1 : t + 1]
            x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
            outputs = model(x)
            comps = compute_score_components(outputs, x, alpha_pred=alpha_pred, beta_recon=beta_recon)
            outputs = attach_window_repr_from_components(model, outputs, comps)
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
            out["window_repr_seq"].append(
                outputs["window_repr"].squeeze(0).detach().cpu().numpy() if outputs.get("window_repr") is not None else None
            )

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


def export_latent_audit_outputs(
    labels,
    dist_fused_to_proto,
    dist_corrected_to_proto,
    correction_norm,
    total_raw_score,
    total_corrected_score,
):
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/latent_audit.csv"
    json_path = "outputs/latent_audit_summary.json"

    audit_df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(labels), dtype=np.int64),
            "label": labels.astype(np.int64),
            "dist_fused_to_proto": dist_fused_to_proto,
            "dist_corrected_to_proto": dist_corrected_to_proto,
            "correction_norm": correction_norm,
            "total_raw_score": total_raw_score,
            "total_corrected_score": total_corrected_score,
        }
    )
    audit_df.to_csv(csv_path, index=False)

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    def _safe_group_mean(values, mask):
        if np.sum(mask) == 0:
            return None
        return float(np.nanmean(values[mask]))

    normal_mean_fused = _safe_group_mean(dist_fused_to_proto, normal_mask)
    anomaly_mean_fused = _safe_group_mean(dist_fused_to_proto, anomaly_mask)
    normal_mean_corrected = _safe_group_mean(dist_corrected_to_proto, normal_mask)
    anomaly_mean_corrected = _safe_group_mean(dist_corrected_to_proto, anomaly_mask)
    normal_mean_corr_norm = _safe_group_mean(correction_norm, normal_mask)
    anomaly_mean_corr_norm = _safe_group_mean(correction_norm, anomaly_mask)

    fused_gap = None
    corrected_gap = None
    gap_change = None
    if (anomaly_mean_fused is not None) and (normal_mean_fused is not None):
        fused_gap = float(anomaly_mean_fused - normal_mean_fused)
    if (anomaly_mean_corrected is not None) and (normal_mean_corrected is not None):
        corrected_gap = float(anomaly_mean_corrected - normal_mean_corrected)
    if (fused_gap is not None) and (corrected_gap is not None):
        gap_change = float(corrected_gap - fused_gap)

    summary = {
        "normal_mean_dist_fused_to_proto": normal_mean_fused,
        "anomaly_mean_dist_fused_to_proto": anomaly_mean_fused,
        "normal_mean_dist_corrected_to_proto": normal_mean_corrected,
        "anomaly_mean_dist_corrected_to_proto": anomaly_mean_corrected,
        "normal_mean_correction_norm": normal_mean_corr_norm,
        "anomaly_mean_correction_norm": anomaly_mean_corr_norm,
        "means": {
            "normal_dist_fused_to_proto": normal_mean_fused,
            "anomaly_dist_fused_to_proto": anomaly_mean_fused,
            "normal_dist_corrected_to_proto": normal_mean_corrected,
            "anomaly_dist_corrected_to_proto": anomaly_mean_corrected,
            "normal_correction_norm": normal_mean_corr_norm,
            "anomaly_correction_norm": anomaly_mean_corr_norm,
        },
        "judgement": {
            "corrected_closer_than_fused_on_normal": (
                bool(normal_mean_corrected < normal_mean_fused)
                if (normal_mean_corrected is not None and normal_mean_fused is not None)
                else None
            ),
            "corrected_farther_than_fused_on_anomaly": (
                bool(anomaly_mean_corrected > anomaly_mean_fused)
                if (anomaly_mean_corrected is not None and anomaly_mean_fused is not None)
                else None
            ),
            "fused_gap_anomaly_minus_normal": fused_gap,
            "corrected_gap_anomaly_minus_normal": corrected_gap,
            "gap_change_corrected_minus_fused": gap_change,
            "separation_enhanced": (bool(gap_change > 0.0) if gap_change is not None else None),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Latent audit summary")
    print(f"normal mean dist_fused_to_proto: {normal_mean_fused}")
    print(f"anomaly mean dist_fused_to_proto: {anomaly_mean_fused}")
    print(f"normal mean dist_corrected_to_proto: {normal_mean_corrected}")
    print(f"anomaly mean dist_corrected_to_proto: {anomaly_mean_corrected}")
    print(f"normal mean correction_norm: {normal_mean_corr_norm}")
    print(f"anomaly mean correction_norm: {anomaly_mean_corr_norm}")
    print(f"Saved latent audit csv: {csv_path}")
    print(f"Saved latent audit summary json: {json_path}")


def export_prototype_path_audit(config, raw_diag, corrected_diag, hybrid_diag, audit_buffers):
    os.makedirs("outputs", exist_ok=True)
    audit_path = "outputs/prototype_path_audit.json"

    proto_cfg = config.get("model", {}).get("prototype", {})
    switches = {
        "use_prototype_fusion": bool(config.get("model", {}).get("use_prototype_fusion", False)),
        "use_node_prototype": bool(proto_cfg.get("use_node_prototype", True)),
        "use_patch_prototype": bool(proto_cfg.get("use_patch_prototype", True)),
        "use_node_correction": bool(proto_cfg.get("use_node_correction", True)),
        "use_patch_correction": bool(proto_cfg.get("use_patch_correction", True)),
    }

    node_usage = None
    if len(audit_buffers["node_usage"]) > 0:
        node_usage = np.mean(np.stack(audit_buffers["node_usage"], axis=0), axis=0).tolist()
    patch_usage = None
    if len(audit_buffers["patch_usage"]) > 0:
        patch_usage = np.mean(np.stack(audit_buffers["patch_usage"], axis=0), axis=0).tolist()

    patch_disabled = not bool(proto_cfg.get("use_patch_prototype", True))

    summary = {
        "switches": switches,
        "means": {
            "node_delta_norm": float(np.mean(audit_buffers["node_delta_norm"])) if audit_buffers["node_delta_norm"] else 0.0,
            "patch_delta_norm": (None if patch_disabled else float(np.mean(audit_buffers["patch_delta_norm"])))
            if audit_buffers["patch_delta_norm"]
            else (None if patch_disabled else 0.0),
            "correction_norm": float(np.mean(audit_buffers["correction_norm"])) if audit_buffers["correction_norm"] else 0.0,
            "node_assignment_entropy": float(np.mean(audit_buffers["node_entropy"])) if audit_buffers["node_entropy"] else 0.0,
            "patch_assignment_entropy": (None if patch_disabled else float(np.mean(audit_buffers["patch_entropy"])))
            if audit_buffers["patch_entropy"]
            else (None if patch_disabled else 0.0),
        },
        "node_delta_norm_mean": float(np.mean(audit_buffers["node_delta_norm"])) if audit_buffers["node_delta_norm"] else 0.0,
        "patch_delta_norm_mean": (None if patch_disabled else float(np.mean(audit_buffers["patch_delta_norm"])))
        if audit_buffers["patch_delta_norm"]
        else (None if patch_disabled else 0.0),
        "correction_norm_mean": float(np.mean(audit_buffers["correction_norm"])) if audit_buffers["correction_norm"] else 0.0,
        "node_assignment_entropy": float(np.mean(audit_buffers["node_entropy"])) if audit_buffers["node_entropy"] else 0.0,
        "patch_assignment_entropy": (None if patch_disabled else float(np.mean(audit_buffers["patch_entropy"])))
        if audit_buffers["patch_entropy"]
        else (None if patch_disabled else 0.0),
        "patch_path_status": "disabled" if patch_disabled else "enabled",
        "usage_frequency": {
            "node_prototype_usage": node_usage,
            "patch_prototype_usage": None if patch_disabled else patch_usage,
        },
        "node_prototype_usage": node_usage,
        "patch_prototype_usage": [] if patch_disabled else (patch_usage or []),
        "slot_level_audit": audit_buffers.get("slot_level", {}),
        "branch_results": {
            "raw": {
                "auc": float(raw_diag["metrics"]["auc"]),
                "best_f1": float(raw_diag["metrics"]["best_f1"]),
                "f1_pa": float(raw_diag["metrics"]["f1_pa"]),
                "dual_threshold_f1": float(raw_diag["dual_f1"]),
                "high_threshold": float(raw_diag["high"]),
                "low_threshold": float(raw_diag["low"]),
            },
            "corrected": {
                "auc": float(corrected_diag["metrics"]["auc"]),
                "best_f1": float(corrected_diag["metrics"]["best_f1"]),
                "f1_pa": float(corrected_diag["metrics"]["f1_pa"]),
                "dual_threshold_f1": float(corrected_diag["dual_f1"]),
                "high_threshold": float(corrected_diag["high"]),
                "low_threshold": float(corrected_diag["low"]),
            },
            "hybrid": {
                "auc": float(hybrid_diag["metrics"]["auc"]),
                "best_f1": float(hybrid_diag["metrics"]["best_f1"]),
                "f1_pa": float(hybrid_diag["metrics"]["f1_pa"]),
                "dual_threshold_f1": float(hybrid_diag["dual_f1"]),
                "high_threshold": float(hybrid_diag["high"]),
                "low_threshold": float(hybrid_diag["low"]),
            },
        },
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved prototype path audit json: {audit_path}")


def _slot_entropy(slot_weights):
    if slot_weights is None:
        return None
    w = np.asarray(slot_weights, dtype=np.float32)
    if w.size == 0:
        return None
    w = np.clip(w, 1e-8, 1.0)
    entropy = -(w * np.log(w)).sum(axis=2)
    return float(np.mean(entropy))


def _slot_diff(slots):
    if slots is None:
        return None
    arr = np.asarray(slots, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[2] < 2:
        return 0.0
    s = arr.shape[2]
    if s == 2:
        diff = np.linalg.norm(arr[:, :, 0, :] - arr[:, :, 1, :], axis=-1)
        return float(np.mean(diff))
    pair_diffs = []
    for i in range(s):
        for j in range(i + 1, s):
            pair_diffs.append(np.linalg.norm(arr[:, :, i, :] - arr[:, :, j, :], axis=-1))
    if not pair_diffs:
        return 0.0
    return float(np.mean(np.stack(pair_diffs, axis=0)))


def export_slot_gate_audit(config, slot_gate_buffers, labels=None):
    os.makedirs("outputs", exist_ok=True)
    audit_path = "outputs/slot_gate_audit.json"
    model_cfg = config.get("model", {})
    if not bool(model_cfg.get("return_slot_debug", False)):
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump({"warning": "return_slot_debug=false, slot/gate audit skipped."}, f, ensure_ascii=False, indent=2)
        print(f"Saved slot gate audit json: {audit_path}")
        return

    local_w = slot_gate_buffers.get("local_slot_weights")
    global_w = slot_gate_buffers.get("global_slot_weights")
    local_slots = slot_gate_buffers.get("z_local_slots")
    global_slots = slot_gate_buffers.get("z_global_slots")
    gate_vals = slot_gate_buffers.get("local_global_gate")

    summary = {
        "local_slot_weight_mean": float(np.mean(local_w)) if local_w is not None else None,
        "local_slot_weight_std": float(np.std(local_w)) if local_w is not None else None,
        "global_slot_weight_mean": float(np.mean(global_w)) if global_w is not None else None,
        "global_slot_weight_std": float(np.std(global_w)) if global_w is not None else None,
        "local_slot_entropy_mean": _slot_entropy(local_w),
        "global_slot_entropy_mean": _slot_entropy(global_w),
        "local_global_gate_mean": float(np.mean(gate_vals)) if gate_vals is not None else None,
        "local_global_gate_std": float(np.std(gate_vals)) if gate_vals is not None else None,
        "global_slot_diff_mean": _slot_diff(global_slots),
        "local_slot_diff_mean": _slot_diff(local_slots),
    }

    if labels is not None:
        total = labels.shape[0]
        if local_w is not None and local_w.shape[0] == total:
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            summary["normal_local_slot_weight_mean"] = float(np.mean(local_w[normal_mask])) if np.any(normal_mask) else None
            summary["anomaly_local_slot_weight_mean"] = (
                float(np.mean(local_w[anomaly_mask])) if np.any(anomaly_mask) else None
            )
        if global_w is not None and global_w.shape[0] == total:
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            summary["normal_global_slot_weight_mean"] = (
                float(np.mean(global_w[normal_mask])) if np.any(normal_mask) else None
            )
            summary["anomaly_global_slot_weight_mean"] = (
                float(np.mean(global_w[anomaly_mask])) if np.any(anomaly_mask) else None
            )
        if gate_vals is not None and gate_vals.shape[0] == total:
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            summary["normal_local_global_gate_mean"] = (
                float(np.mean(gate_vals[normal_mask])) if np.any(normal_mask) else None
            )
            summary["anomaly_local_global_gate_mean"] = (
                float(np.mean(gate_vals[anomaly_mask])) if np.any(anomaly_mask) else None
            )
        if global_slots is not None and global_slots.shape[0] == total:
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            summary["normal_global_slot_diff_mean"] = _slot_diff(global_slots[normal_mask]) if np.any(normal_mask) else None
            summary["anomaly_global_slot_diff_mean"] = (
                _slot_diff(global_slots[anomaly_mask]) if np.any(anomaly_mask) else None
            )

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved slot gate audit json: {audit_path}")


def compute_recon_loss(recon, x, criterion_mse, train_cfg):
    recon_mode = str(train_cfg.get("recon_mode", "full_window")).lower()
    lambda_recon_last = float(train_cfg.get("lambda_recon_last", 1.0))
    lambda_recon_full = float(train_cfg.get("lambda_recon_full", 0.2))

    if recon_mode == "last_point":
        l_recon = criterion_mse(recon[:, -1, :], x[:, -1, :])
    elif recon_mode == "mixed":
        l_recon_last = criterion_mse(recon[:, -1, :], x[:, -1, :])
        l_recon_full = criterion_mse(recon, x)
        l_recon = lambda_recon_last * l_recon_last + lambda_recon_full * l_recon_full
    else:
        recon_mode = "full_window"
        l_recon = criterion_mse(recon, x)

    return l_recon, recon_mode, lambda_recon_last, lambda_recon_full


def compute_slot_proto_stats(prototype_fusion, z_slots_np, labels=None):
    if prototype_fusion is None or z_slots_np is None:
        return {}
    if not hasattr(prototype_fusion, "match_to_node_prototypes"):
        return {}

    with torch.no_grad():
        z_slots = torch.from_numpy(z_slots_np).to(DEVICE)
        b, n, s, d = z_slots.shape
        z_flat = z_slots.reshape(b, n * s, d)
        _, assign, delta = prototype_fusion.match_to_node_prototypes(z_flat)
        if assign is None or delta is None:
            return {}
        dist = torch.norm(delta, p=2, dim=-1).reshape(b, n, s).mean(dim=(1, 2)).detach().cpu().numpy()
        entropy = (-(assign * torch.log(assign + 1e-8)).sum(dim=-1)).reshape(b, n, s).mean(dim=(1, 2)).detach().cpu().numpy()

    out = {
        "distance_mean": float(np.mean(dist)),
        "entropy_mean": float(np.mean(entropy)),
    }
    if labels is not None and len(labels) == len(dist):
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        out["normal_distance_mean"] = float(np.mean(dist[normal_mask])) if np.any(normal_mask) else None
        out["anomaly_distance_mean"] = float(np.mean(dist[anomaly_mask])) if np.any(anomaly_mask) else None
    return out


def evaluate_subset_for_checkpoint(model, test_loader, labels, config, max_batches=20):
    infer_cfg = config.get("inference", {})
    alpha_pred = float(infer_cfg.get("alpha_pred", 1.0))
    beta_recon = float(infer_cfg.get("beta_recon", 1.0))

    corrected_scores = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            x = batch.to(DEVICE)
            outputs = model(x)
            pred_corr = outputs["pred_corrected"]
            recon_corr = outputs["recon_corrected"]
            pred_corr_score = torch.mean((pred_corr - x[:, -1, :]) ** 2, dim=1)
            recon_corr_last_score = torch.mean((recon_corr[:, -1, :] - x[:, -1, :]) ** 2, dim=1)
            total_corrected_score = alpha_pred * pred_corr_score + beta_recon * recon_corr_last_score
            corrected_scores.append(total_corrected_score.detach().cpu().numpy())

    if not corrected_scores:
        return None

    corrected_scores = np.concatenate(corrected_scores).astype(np.float32)
    if labels is None or len(labels) == 0:
        return None
    valid_len = min(len(corrected_scores), len(labels))
    corrected_scores = corrected_scores[:valid_len]
    labels = labels[:valid_len]
    if len(np.unique(labels)) < 2:
        return None

    metrics = get_best_f1(labels, corrected_scores)
    return {
        "auc": float(metrics["auc"]),
        "pa_f1": float(metrics["f1_pa"]),
    }


def train(args):
    config = load_config(args.config)
    print(f"Mode: TRAIN | Device: {DEVICE}")
    print(f"Config: {args.config}")

    train_loader, test_loader, input_dim = get_dataloaders(args.config)
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
    train_cfg = config.get("train", {})
    configured_recon_mode = str(train_cfg.get("recon_mode", "full_window")).lower()
    lambda_recon_last = float(train_cfg.get("lambda_recon_last", 1.0))
    lambda_recon_full = float(train_cfg.get("lambda_recon_full", 0.2))
    checkpoint_mode = str(train_cfg.get("checkpoint_mode", "loss")).lower()
    val_eval_interval = int(train_cfg.get("val_eval_interval", 5))
    val_subset_max_batches = int(train_cfg.get("val_subset_max_batches", 20))
    lambda_window_repr_cl = float(train_cfg.get("lambda_window_repr_cl", 0.05))
    if checkpoint_mode not in ("loss", "pa_f1", "auc"):
        checkpoint_mode = "loss"

    labels = load_labels(config)

    print("Start Training")
    print(
        f"Reconstruction mode: {configured_recon_mode} | "
        f"lambda_recon_last={lambda_recon_last:.4f} | "
        f"lambda_recon_full={lambda_recon_full:.4f}"
    )
    print(
        f"Checkpoint mode: {checkpoint_mode} | "
        f"val_eval_interval={val_eval_interval} | "
        f"val_subset_max_batches={val_subset_max_batches}"
    )
    model.train()
    best_loss_path = "best_model_loss.pth"
    best_pa_f1_path = "best_model_pa_f1.pth"
    best_auc_path = "best_model_auc.pth"
    best_pa_f1 = -1.0
    best_auc = -1.0

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
            l_recon, active_recon_mode, lambda_recon_last, lambda_recon_full = compute_recon_loss(
                recon=recon,
                x=x,
                criterion_mse=criterion_mse,
                train_cfg=train_cfg,
            )

            noise = torch.randn_like(x) * 0.01
            outputs_aug = model(x + noise)

            z1 = outputs["z_local"].reshape(x.size(0), -1)
            z2 = outputs_aug["z_local"].reshape(x.size(0), -1)
            l_cl = criterion_cl(z1, z2)

            comps_clean = compute_score_components(outputs, x)
            outputs = attach_window_repr_from_components(model, outputs, comps_clean)
            l_window_repr = torch.tensor(0.0, device=DEVICE)
            if outputs.get("window_repr") is not None:
                comps_aug = compute_score_components(outputs_aug, x + noise)
                outputs_aug = attach_window_repr_from_components(model, outputs_aug, comps_aug)
                if outputs_aug.get("window_repr") is not None:
                    l_window_repr = torch.mean((outputs["window_repr"] - outputs_aug["window_repr"]) ** 2)

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
                + lambda_window_repr_cl * l_window_repr
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
            torch.save(model.state_dict(), best_loss_path)
            print(f"Saved Best Model ({avg_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping Triggered")
                break

        if val_eval_interval > 0 and ((epoch + 1) % val_eval_interval == 0):
            val_metrics = evaluate_subset_for_checkpoint(
                model=model,
                test_loader=test_loader,
                labels=labels,
                config=config,
                max_batches=val_subset_max_batches,
            )
            if val_metrics is not None:
                current_pa_f1 = val_metrics["pa_f1"]
                current_auc = val_metrics["auc"]
                print(f"Validation subset metrics | PA-F1: {current_pa_f1:.4f} | AUC: {current_auc:.4f}")
                if current_pa_f1 > best_pa_f1:
                    best_pa_f1 = current_pa_f1
                    torch.save(model.state_dict(), best_pa_f1_path)
                if current_auc > best_auc:
                    best_auc = current_auc
                    torch.save(model.state_dict(), best_auc_path)
            model.train()

    print(f"Training Complete. Model saved to {save_path}")
    source_checkpoint = best_loss_path
    if checkpoint_mode == "pa_f1" and os.path.exists(best_pa_f1_path):
        source_checkpoint = best_pa_f1_path
    elif checkpoint_mode == "auc" and os.path.exists(best_auc_path):
        source_checkpoint = best_auc_path
    elif not os.path.exists(best_loss_path):
        source_checkpoint = save_path
    if os.path.exists(source_checkpoint):
        shutil.copyfile(source_checkpoint, save_path)

    os.makedirs("outputs", exist_ok=True)
    summary_path = "outputs/train_loss_summary.json"
    train_summary = {
        "recon_mode": active_recon_mode if "active_recon_mode" in locals() else configured_recon_mode,
        "lambda_recon_last": float(lambda_recon_last),
        "lambda_recon_full": float(lambda_recon_full),
        "lambda_window_repr_cl": float(lambda_window_repr_cl),
        "best_loss": float(best_loss),
        "checkpoint_mode": checkpoint_mode,
        "val_eval_interval": int(val_eval_interval),
        "val_subset_max_batches": int(val_subset_max_batches),
        "best_pa_f1": None if best_pa_f1 < 0 else float(best_pa_f1),
        "best_auc": None if best_auc < 0 else float(best_auc),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved train loss summary: {summary_path}")


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
    dist_fused_to_proto = []
    dist_corrected_to_proto = []
    correction_norm_list = []
    total_raw_score_list = []
    total_corrected_score_list = []
    prototype_audit_buffers = {
        "node_delta_norm": [],
        "patch_delta_norm": [],
        "correction_norm": [],
        "node_usage": [],
        "patch_usage": [],
        "node_entropy": [],
        "patch_entropy": [],
        "slot_level": {},
    }
    slot_gate_buffers = {
        "local_slot_weights": [],
        "global_slot_weights": [],
        "z_local_slots": [],
        "z_global_slots": [],
        "local_global_gate": [],
    }

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
            correction_norm_list.append(comps["correction_norm"].detach().cpu().numpy())
            total_raw_score_list.append(comps["total_raw_score"].detach().cpu().numpy())
            total_corrected_score_list.append(comps["total_corrected_score"].detach().cpu().numpy())

            z_fused = outputs.get("z_fused")
            z_corrected = outputs.get("z_corrected")
            node_proto_latent = outputs.get("node_proto_latent")
            if z_fused is not None and z_corrected is not None and node_proto_latent is not None:
                dist_fused = torch.mean((z_fused - node_proto_latent) ** 2, dim=(1, 2))
                dist_corrected = torch.mean((z_corrected - node_proto_latent) ** 2, dim=(1, 2))
                dist_fused_to_proto.append(dist_fused.detach().cpu().numpy())
                dist_corrected_to_proto.append(dist_corrected.detach().cpu().numpy())
            else:
                batch_size = x.shape[0]
                dist_fused_to_proto.append(np.full((batch_size,), np.nan, dtype=np.float32))
                dist_corrected_to_proto.append(np.full((batch_size,), np.nan, dtype=np.float32))

            node_delta = outputs.get("node_delta")
            if node_delta is not None:
                node_delta_norm = torch.mean(node_delta ** 2, dim=(1, 2))
                prototype_audit_buffers["node_delta_norm"].append(float(node_delta_norm.mean().item()))
            else:
                prototype_audit_buffers["node_delta_norm"].append(0.0)

            patch_delta = outputs.get("patch_delta")
            if patch_delta is not None:
                patch_delta_norm = torch.mean(patch_delta ** 2, dim=(1, 2))
                prototype_audit_buffers["patch_delta_norm"].append(float(patch_delta_norm.mean().item()))
            else:
                prototype_audit_buffers["patch_delta_norm"].append(0.0)

            prototype_audit_buffers["correction_norm"].append(float(comps["correction_norm"].mean().item()))

            node_assign = outputs.get("node_assign")
            if node_assign is not None:
                prototype_audit_buffers["node_usage"].append(
                    node_assign.mean(dim=(0, 1)).detach().cpu().numpy()
                )
                node_entropy_batch = -(node_assign * torch.log(node_assign + 1e-8)).sum(dim=-1).mean(dim=1)
                prototype_audit_buffers["node_entropy"].append(float(node_entropy_batch.mean().item()))
            else:
                prototype_audit_buffers["node_entropy"].append(0.0)

            patch_assign = outputs.get("patch_assign")
            if patch_assign is not None:
                prototype_audit_buffers["patch_usage"].append(
                    patch_assign.mean(dim=(0, 1)).detach().cpu().numpy()
                )
                patch_entropy_batch = -(patch_assign * torch.log(patch_assign + 1e-8)).sum(dim=-1).mean(dim=1)
                prototype_audit_buffers["patch_entropy"].append(float(patch_entropy_batch.mean().item()))
            else:
                prototype_audit_buffers["patch_entropy"].append(0.0)

            z_local_slots = outputs.get("z_local_slots")
            if z_local_slots is not None:
                slot_gate_buffers["z_local_slots"].append(z_local_slots.detach().cpu().numpy())
            z_global_slots = outputs.get("z_global_slots")
            if z_global_slots is not None:
                slot_gate_buffers["z_global_slots"].append(z_global_slots.detach().cpu().numpy())
            slot_w_local = outputs.get("slot_weights_local")
            if slot_w_local is not None:
                slot_gate_buffers["local_slot_weights"].append(slot_w_local.detach().cpu().numpy())
            slot_w_global = outputs.get("slot_weights_global")
            if slot_w_global is not None:
                slot_gate_buffers["global_slot_weights"].append(slot_w_global.detach().cpu().numpy())
            lg_gate = outputs.get("local_global_gate")
            if lg_gate is not None:
                slot_gate_buffers["local_global_gate"].append(lg_gate.detach().cpu().numpy())

    raw_scores = np.concatenate(raw_scores)
    corrected_scores = np.concatenate(corrected_scores)
    hybrid_scores = np.concatenate(hybrid_scores)
    dist_fused_to_proto = np.concatenate(dist_fused_to_proto).astype(np.float32)
    dist_corrected_to_proto = np.concatenate(dist_corrected_to_proto).astype(np.float32)
    correction_norm_list = np.concatenate(correction_norm_list).astype(np.float32)
    total_raw_score_list = np.concatenate(total_raw_score_list).astype(np.float32)
    total_corrected_score_list = np.concatenate(total_corrected_score_list).astype(np.float32)

    labels = load_labels(config)
    if labels is None:
        print("No labels found. Skipping evaluation metrics.")
        return

    min_len = min(
        len(raw_scores),
        len(corrected_scores),
        len(hybrid_scores),
        len(labels),
        len(dist_fused_to_proto),
        len(dist_corrected_to_proto),
        len(correction_norm_list),
        len(total_raw_score_list),
        len(total_corrected_score_list),
    )
    raw_scores = raw_scores[:min_len]
    corrected_scores = corrected_scores[:min_len]
    hybrid_scores = hybrid_scores[:min_len]
    labels = labels[:min_len]
    dist_fused_to_proto = dist_fused_to_proto[:min_len]
    dist_corrected_to_proto = dist_corrected_to_proto[:min_len]
    correction_norm_list = correction_norm_list[:min_len]
    total_raw_score_list = total_raw_score_list[:min_len]
    total_corrected_score_list = total_corrected_score_list[:min_len]

    for k in list(slot_gate_buffers.keys()):
        if slot_gate_buffers[k]:
            merged = np.concatenate(slot_gate_buffers[k], axis=0)
            slot_gate_buffers[k] = merged[:min_len]
        else:
            slot_gate_buffers[k] = None

    local_slot_stats = compute_slot_proto_stats(
        model.prototype_fusion if hasattr(model, "prototype_fusion") else None,
        slot_gate_buffers["z_local_slots"],
        labels=labels,
    )
    global_slot_stats = compute_slot_proto_stats(
        model.prototype_fusion if hasattr(model, "prototype_fusion") else None,
        slot_gate_buffers["z_global_slots"],
        labels=labels,
    )
    prototype_audit_buffers["slot_level"] = {
        "local_slot_proto_distance_mean": local_slot_stats.get("distance_mean"),
        "normal_local_slot_proto_distance_mean": local_slot_stats.get("normal_distance_mean"),
        "anomaly_local_slot_proto_distance_mean": local_slot_stats.get("anomaly_distance_mean"),
        "global_slot_proto_distance_mean": global_slot_stats.get("distance_mean"),
        "normal_global_slot_proto_distance_mean": global_slot_stats.get("normal_distance_mean"),
        "anomaly_global_slot_proto_distance_mean": global_slot_stats.get("anomaly_distance_mean"),
        "local_slot_proto_entropy_mean": local_slot_stats.get("entropy_mean"),
        "global_slot_proto_entropy_mean": global_slot_stats.get("entropy_mean"),
    }

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

    export_latent_audit_outputs(
        labels=labels,
        dist_fused_to_proto=dist_fused_to_proto,
        dist_corrected_to_proto=dist_corrected_to_proto,
        correction_norm=correction_norm_list,
        total_raw_score=total_raw_score_list,
        total_corrected_score=total_corrected_score_list,
    )
    export_prototype_path_audit(
        config=config,
        raw_diag=raw_diag,
        corrected_diag=corrected_diag,
        hybrid_diag=hybrid_diag,
        audit_buffers=prototype_audit_buffers,
    )
    export_slot_gate_audit(
        config=config,
        slot_gate_buffers=slot_gate_buffers,
        labels=labels,
    )

    if bool(anomaly_space_cfg.get("enable_memory_bank", True)) and bool(infer_cfg.get("save_anomaly_segments", True)):
        segment_repr_mode = anomaly_space_cfg.get("segment_repr_mode", "static_concat")
        memory_bank = AnomalyMemoryBank(segment_repr_mode=segment_repr_mode)
        min_persistence = int(infer_cfg.get("min_anomaly_persistence", 3))

        schema = SegmentEmbeddingSchema(
            latent_dim=int(config.get("model", {}).get("hidden_dim", 64)),
            num_node_prototypes=int(config.get("model", {}).get("prototype", {}).get("num_node_prototypes", 8)),
            num_patch_prototypes=int(config.get("model", {}).get("prototype", {}).get("num_patch_prototypes", 8)),
            segment_repr_dim=int(anomaly_space_cfg.get("segment_repr_dim", 64)),
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
                    window_repr_slice = [w for w in rollout["window_repr_seq"][start_idx : end_idx + 1] if w is not None]
                    if window_repr_slice:
                        window_seq = np.stack(window_repr_slice, axis=0)
                        seg_repr = build_learnable_segment_repr(
                            window_seq,
                            repr_dim=int(anomaly_space_cfg.get("segment_repr_dim", 64)),
                            pool_mode=anomaly_space_cfg.get("segment_pool_mode", "attentive"),
                        )
                        record["window_repr_seq"] = window_seq.tolist()
                        record["segment_repr"] = seg_repr.tolist()
                    else:
                        record["window_repr_seq"] = None
                        record["segment_repr"] = None
                    memory_bank.append_segment(record, schema=schema, segment_repr_mode=segment_repr_mode)
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
