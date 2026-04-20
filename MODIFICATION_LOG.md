# 修改日志

更新时间：2026-04-20

## 1) 指标计算修复（`src/utils/metrics.py`）
- 修复 `get_best_f1` 中阈值索引对齐问题：
  - `precision_recall_curve` 返回的 `precision/recall` 长度比 `thresholds` 多 1；
  - 旧逻辑直接对完整 `f1_scores` 做 `argmax` 可能导致阈值索引越界或错位；
  - 新逻辑使用与 `thresholds` 对齐的 `f1_scores[:-1]` 计算最佳索引。
- 增加 `thresholds.size == 0` 的兜底分支，确保在极端输入时也能返回有效阈值。
- 统一返回值中的 `best_f1` 与 `threshold` 为 `float`。

## 2) 新增单元测试（`aiops_test_unit/test_metrics_utils.py`）
- 新增针对 `get_best_f1` 的边界测试：
  - 最优点在尾部附近时不发生索引问题；
  - 单类别输入场景下不触发阈值索引异常（允许 `roc_auc_score` 的既有行为）。

## 3) 配置读取编码兼容性修复
### `main.py`
- `load_config` 改为 `encoding='utf-8-sig'` 读取 YAML，兼容 Windows 默认编码环境与 BOM。

### `src/data/loader.py`
- `get_dataloaders` 读取配置时同样改为 `encoding='utf-8-sig'`，避免路径切换时再次触发解码错误。

## 4) 测试流程稳定性修复（`main.py`）
- 修复 `evaluate` 中 dataloader 解包问题：正确获取 `test_loader`。
- 增加阈值初始化逻辑：
  - 优先读取 `config.inference.high_threshold/low_threshold`；
  - 若缺失，则基于训练集参考分数自动计算双阈值（`compute_reference_thresholds`）。

## 5) 日志输出规范化（移除表情符号）
### `main.py`
- 移除所有 `print(...)` 输出中的表情符号，保留原语义与流程。

### `src/data/loader.py`
- 移除所有 `print(...)` 输出中的表情符号，保留原语义与流程。

## 6) 已执行的基础校验
- 语法校验：`python -m py_compile main.py src/data/loader.py` 通过。
- 关键词检查：对 `main.py` 与 `src/data/loader.py` 的 `print(...)` 输出进行扫描，确认已去除表情符号。

