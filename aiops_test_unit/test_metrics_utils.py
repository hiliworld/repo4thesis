import numpy as np

from src.utils.metrics import get_best_f1


def test_get_best_f1_no_index_error_when_best_at_tail():
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)

    result = get_best_f1(labels, scores)

    assert "threshold" in result
    assert np.isfinite(result["best_f1"])
    assert np.isfinite(result["threshold"])


def test_get_best_f1_handles_single_class_without_thresholds():
    labels = np.array([0, 0, 0, 0], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    try:
        result = get_best_f1(labels, scores)
    except ValueError:
        # roc_auc_score 在单类别输入时会抛 ValueError，确认不是阈值索引问题
        return

    assert np.isfinite(result["threshold"])
