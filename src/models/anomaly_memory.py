import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans


@dataclass
class SegmentEmbeddingSchema:
    latent_dim: int
    num_node_prototypes: int
    num_patch_prototypes: int
    segment_repr_dim: int = 64


def _to_1d_array(value, default_len=1):
    if value is None:
        return np.zeros(default_len, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(default_len, dtype=np.float32)
    return arr


def build_segment_embedding(segment_record: Dict, schema: Optional[SegmentEmbeddingSchema] = None) -> np.ndarray:
    """Static concat fallback embedding."""
    if schema is None:
        zc = _to_1d_array(segment_record.get("z_corrected_mean"), default_len=1)
        zf = _to_1d_array(segment_record.get("z_fused_mean"), default_len=zc.size)
        nh = _to_1d_array(segment_record.get("node_assign_hist"), default_len=1)
        ph = _to_1d_array(segment_record.get("patch_assign_hist"), default_len=1)
    else:
        zc = _to_1d_array(segment_record.get("z_corrected_mean"), default_len=schema.latent_dim)
        zf = _to_1d_array(segment_record.get("z_fused_mean"), default_len=schema.latent_dim)
        nh = _to_1d_array(segment_record.get("node_assign_hist"), default_len=schema.num_node_prototypes)
        ph = _to_1d_array(segment_record.get("patch_assign_hist"), default_len=schema.num_patch_prototypes)

    score_stats = np.asarray(
        [
            float(segment_record.get("peak_score", 0.0)),
            float(segment_record.get("mean_score", 0.0)),
            float(segment_record.get("length", 0.0)),
        ],
        dtype=np.float32,
    )
    delta_stats = np.asarray(
        [
            float(segment_record.get("node_delta_mean", 0.0)),
            float(segment_record.get("patch_delta_mean", 0.0)),
            float(segment_record.get("correction_norm_mean", 0.0)),
        ],
        dtype=np.float32,
    )

    return np.concatenate([zc, zf, score_stats, nh, ph, delta_stats], axis=0).astype(np.float32)


def build_learnable_segment_repr(window_repr_seq, repr_dim=64, pool_mode="attentive"):
    """Numpy implementation for test-time fallback segment representation.

    Args:
        window_repr_seq: [L, E]
        repr_dim: fallback output dim when empty
        pool_mode: attentive / mean
    Returns:
        segment_repr: [E]
    """
    seq = np.asarray(window_repr_seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    if seq.size == 0:
        return np.zeros((int(repr_dim),), dtype=np.float32)

    mode = str(pool_mode or "attentive").lower()
    if mode == "mean" or seq.shape[0] == 1:
        vec = seq.mean(axis=0)
        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).astype(np.float32)

    q = seq.mean(axis=0, keepdims=True)  # [1, E]
    logits = np.matmul(seq, q.T).reshape(-1)  # [L]
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weights = weights / (np.sum(weights) + 1e-8)
    vec = np.sum(seq * weights[:, None], axis=0)
    norm = np.linalg.norm(vec) + 1e-8
    return (vec / norm).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def retrieve_anomaly_pattern(segment_embedding, cluster_centers, unknown_similarity_threshold=0.55):
    if cluster_centers is None or len(cluster_centers) == 0:
        return {
            "nearest_cluster_id": -1,
            "nearest_similarity": 0.0,
            "is_unknown": True,
        }

    emb = np.asarray(segment_embedding, dtype=np.float32).reshape(-1)
    similarities = []
    cluster_ids = []
    for cid, center in cluster_centers.items():
        center_vec = np.asarray(center, dtype=np.float32).reshape(-1)
        similarities.append(_cosine_similarity(emb, center_vec))
        cluster_ids.append(int(cid))

    best_idx = int(np.argmax(similarities))
    nearest_similarity = float(similarities[best_idx])
    nearest_cluster_id = int(cluster_ids[best_idx])
    return {
        "nearest_cluster_id": nearest_cluster_id,
        "nearest_similarity": nearest_similarity,
        "is_unknown": bool(nearest_similarity < float(unknown_similarity_threshold)),
    }


class AnomalyMemoryBank:
    def __init__(self, segment_repr_mode="static_concat"):
        self.records: List[Dict] = []
        self.cluster_metadata: List[Dict] = []
        self.segment_repr_mode = str(segment_repr_mode or "static_concat").lower()

    def append_segment(
        self,
        segment_record: Dict,
        schema: Optional[SegmentEmbeddingSchema] = None,
        segment_repr_mode: Optional[str] = None,
    ):
        rec = dict(segment_record)
        rec.setdefault("cluster_id", -1)
        rec.setdefault("nearest_cluster_id", -1)
        rec.setdefault("nearest_similarity", 0.0)
        rec.setdefault("is_unknown", True)

        mode = str(segment_repr_mode or self.segment_repr_mode).lower()
        rec["segment_repr_mode"] = mode

        use_learnable = mode == "learnable" and rec.get("segment_repr") is not None
        if use_learnable:
            rec["segment_embedding"] = _to_1d_array(
                rec.get("segment_repr"),
                default_len=(schema.segment_repr_dim if schema else 64),
            ).tolist()
        else:
            rec["segment_embedding"] = build_segment_embedding(rec, schema=schema).tolist()

        self.records.append(rec)

    def get_embeddings_matrix(self) -> np.ndarray:
        if not self.records:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([np.asarray(r["segment_embedding"], dtype=np.float32) for r in self.records], axis=0)

    def set_cluster_metadata(self, metadata: List[Dict]):
        self.cluster_metadata = metadata

    def save_pkl(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "records": self.records,
                    "cluster_metadata": self.cluster_metadata,
                    "segment_repr_mode": self.segment_repr_mode,
                },
                f,
            )

    @classmethod
    def load_pkl(cls, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        bank = cls(segment_repr_mode=obj.get("segment_repr_mode", "static_concat"))
        bank.records = obj.get("records", [])
        bank.cluster_metadata = obj.get("cluster_metadata", [])
        return bank

    def export_csv(self, path: str):
        csv_rows = []
        for r in self.records:
            row = dict(r)
            for key in [
                "z_fused_mean",
                "z_corrected_mean",
                "node_assign_hist",
                "patch_assign_hist",
                "window_repr_seq",
                "segment_repr",
                "segment_embedding",
            ]:
                if isinstance(row.get(key), (list, tuple)):
                    row[key] = json.dumps(row[key])
            csv_rows.append(row)
        pd.DataFrame(csv_rows).to_csv(path, index=False)


def cluster_anomaly_segments(memory_bank: AnomalyMemoryBank, method="agglomerative", num_clusters=8, unknown_similarity_threshold=0.55):
    embeddings = memory_bank.get_embeddings_matrix()
    n_segments = embeddings.shape[0]
    if n_segments == 0:
        memory_bank.set_cluster_metadata([])
        return {"cluster_centers": {}, "cluster_metadata": []}

    method = (method or "agglomerative").lower()
    n_clusters = int(max(1, min(num_clusters, n_segments)))

    if method == "kmeans":
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)
    elif method == "hdbscan":
        try:
            import hdbscan  # type: ignore

            labels = hdbscan.HDBSCAN(min_cluster_size=max(2, n_segments // 10)).fit_predict(embeddings)
            if np.all(labels < 0):
                labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
        except Exception:
            labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    else:
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)

    unique_clusters = sorted(set(int(x) for x in labels if int(x) >= 0))
    if not unique_clusters:
        unique_clusters = [0]
        labels = np.zeros((n_segments,), dtype=np.int32)

    centers: Dict[int, List[float]] = {}
    metadata: List[Dict] = []

    for cid in unique_clusters:
        idxs = np.where(labels == cid)[0]
        cluster_emb = embeddings[idxs]
        center = cluster_emb.mean(axis=0)
        centers[int(cid)] = center.astype(np.float32).tolist()

        dists = np.linalg.norm(cluster_emb - center[None, :], axis=1)
        rep_local_idx = int(np.argmin(dists))
        rep_global_idx = int(idxs[rep_local_idx])
        rep_seg_id = memory_bank.records[rep_global_idx].get("segment_id", rep_global_idx)

        metadata.append(
            {
                "cluster_id": int(cid),
                "cluster_center": centers[int(cid)],
                "num_segments": int(len(idxs)),
                "representative_segment_id": int(rep_seg_id),
            }
        )

    for i, rec in enumerate(memory_bank.records):
        rec["cluster_id"] = int(labels[i])
        retrieval = retrieve_anomaly_pattern(
            rec["segment_embedding"],
            centers,
            unknown_similarity_threshold=unknown_similarity_threshold,
        )
        rec.update(retrieval)

    memory_bank.set_cluster_metadata(metadata)
    return {"cluster_centers": centers, "cluster_metadata": metadata}
