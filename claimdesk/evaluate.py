"""家族隔离与时序评估。

评估纪律：
- 同一家族的换码/换版记录不能跨训练与验证两边（family_group 分组）；
- 提供两类外推视角：后来的版本（同家族留出版本）与未见家族（整族留出）；
- 未标注记录不是负样本，指标只在“当时有合格定稿”的记录上计算，
  覆盖率与无意见规模另行报告；
- 指标按声明类与渠道分别给出：误放率、漏收率、自动处理覆盖率，
  不允许只报总体准确率；
- 随机抽核（random_audit，带 selection_probability）支持逆概率加权，
  估计自动发布集合的全目录错误率；定向疑点队列的结果不能代表全目录。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import CLASSES, is_active_at, parse_ts, spec_listed


@dataclass
class Split:
    name: str
    train_ids: list[str]
    eval_ids: list[str]
    note: str


def temporal_split(products: list[dict], holdout_versions=(3,)) -> Split:
    """同家族：低版本训练、高版本评估，模拟“后来的版本”。"""
    holdout = set(holdout_versions)
    train_ids = [p["id"] for p in products if p["version"] not in holdout]
    eval_ids = [p["id"] for p in products if p["version"] in holdout]
    train_families = {p["family"] for p in products if p["id"] in set(train_ids)}
    eval_families = {p["family"] for p in products if p["id"] in set(eval_ids)}
    if train_families & eval_families and not holdout:
        raise AssertionError
    return Split(
        name="temporal_later_versions",
        train_ids=train_ids,
        eval_ids=eval_ids,
        note="同家族低版本训练、高版本评估；家族在两侧共享但记录不共享",
    )


def family_holdout_split(
    products: list[dict], n_families: int = 30, seed: int = 20261124
) -> Split:
    """整族留出：评估家族在训练侧完全不可见。"""
    rng = np.random.default_rng(seed)
    families = sorted({p["family"] for p in products})
    chosen = set(rng.choice(families, size=min(n_families, len(families)), replace=False))
    train_ids = [p["id"] for p in products if p["family"] not in chosen]
    eval_ids = [p["id"] for p in products if p["family"] in chosen]
    return Split(
        name="unseen_families",
        train_ids=train_ids,
        eval_ids=eval_ids,
        note=f"{len(chosen)}个家族整族留出，训练侧零接触",
    )


def labeled_only(ids: list[str], labels: dict[str, set[str]]) -> list[str]:
    """指标口径：只保留当时有合格定稿的记录；其余记为未标注。"""
    return [pid for pid in ids if pid in labels]


def per_class_reliability(
    y_true: np.ndarray, y_pred: np.ndarray, classes=CLASSES
) -> dict[str, dict]:
    """逐声明类的误放率、漏收率、覆盖率与计数。"""
    out: dict[str, dict] = {}
    for j, cls in enumerate(classes):
        true = y_true[:, j]
        pred = y_pred[:, j]
        negatives = true == 0
        positives = true == 1
        false_releases = int(((pred == 1) & negatives).sum())
        misses = int(((pred == 0) & positives).sum())
        auto = int((pred == 1).sum())
        true_positive = int(((pred == 1) & positives).sum())
        out[cls] = {
            "n": int(len(true)),
            "positives": int(positives.sum()),
            "auto_coverage": round(auto / len(true), 4) if len(true) else None,
            "false_release_rate": round(false_releases / auto, 4) if auto else None,
            "false_release_count": false_releases,
            "miss_rate": round(misses / int(positives.sum()), 4) if positives.sum() else None,
            "miss_count": misses,
            "precision": round(true_positive / auto, 4) if auto else None,
            "recall": round(true_positive / int(positives.sum()), 4) if positives.sum() else None,
        }
    return out


def per_channel_reliability(
    records: list[dict], y_true: np.ndarray, y_pred: np.ndarray, classes=CLASSES
) -> dict[str, dict]:
    """渠道 × 声明类的误放/漏收/覆盖；无标注渠道明确标注不可估计。"""
    channels = sorted({r["channel"] for r in records})
    result: dict[str, dict] = {}
    for channel in channels:
        idx = [i for i, r in enumerate(records) if r["channel"] == channel]
        if not idx:
            continue
        sub_true = y_true[idx]
        sub_pred = y_pred[idx]
        if sub_true.shape[0] == 0:
            result[channel] = {"status": "no_records"}
            continue
        labeled_rows = int((sub_true.sum(axis=1) > 0).sum())
        if labeled_rows == 0:
            result[channel] = {
                "status": "no_eligible_labels",
                "n_records": len(idx),
                "note": "该渠道没有合格定稿，无法估计可靠性，自动决策应转人工",
            }
            continue
        result[channel] = {
            "status": "ok",
            "n_records": len(idx),
            "n_labeled": labeled_rows,
            "classes": per_class_reliability(sub_true, sub_pred, classes),
        }
    return result


def threshold_sweep(
    proba: np.ndarray, y_true: np.ndarray, classes=CLASSES, grid=None
) -> dict[str, list[dict]]:
    """每个声明类在阈值网格上的误放率/漏收率/覆盖，供容限选阈。"""
    if grid is None:
        grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    sweep: dict[str, list[dict]] = {}
    for j, cls in enumerate(classes):
        rows = []
        true = y_true[:, j]
        positives = max(int((true == 1).sum()), 1)
        for threshold in grid:
            pred = (proba[:, j] >= threshold).astype(int)
            auto = int((pred == 1).sum())
            fr = int(((pred == 1) & (true == 0)).sum())
            miss = int(((pred == 0) & (true == 1)).sum())
            rows.append({
                "threshold": float(threshold),
                "auto_coverage": round(auto / len(true), 4),
                "false_release_rate": round(fr / auto, 4) if auto else 0.0,
                "false_release_count": fr,
                "miss_rate": round(miss / positives, 4),
                "miss_count": miss,
            })
        sweep[cls] = rows
    return sweep


def ipw_release_error(
    release_rows: list[dict],
    reviews: list[dict],
    products_index: dict[str, dict],
    classes=CLASSES,
) -> dict:
    """用随机抽核结果对“自动发布集合”做逆概率加权错误率估计。

    release_rows: [{product_id, proba, predicted_classes, units_sold}]
    只有 selected_by=random_audit 且带有效 selection_probability 的定稿可入估计；
    定向疑点（suspect_queue）不携带总体代表性，单独计数、不参与加权。
    """
    audit_by_product: dict[str, list[dict]] = {}
    for review in reviews:
        if not is_active_at(review):
            continue
        audit_by_product.setdefault(review["product_id"], []).append(review)

    per_class = {cls: {"weighted": 0.0, "weight_total": 0.0,
                       "audited": 0, "errors": 0} for cls in classes}
    suspect_audits = 0
    for row in release_rows:
        pid = row["product_id"]
        matched = audit_by_product.get(pid, [])
        random_reviews = [
            rv for rv in matched
            if rv.get("selected_by") == "random_audit"
            and rv.get("selection_probability")
            and spec_listed(rv, pid)
        ]
        if not random_reviews:
            if any(rv.get("selected_by") == "suspect_queue" for rv in matched):
                suspect_audits += 1
            continue
        truth = {cls for rv in random_reviews for cls in rv.get("classes", [])}
        weight = 1.0 / float(random_reviews[0]["selection_probability"])
        for cls in row.get("predicted_classes", []):
            bucket = per_class[cls]
            bucket["weighted"] += weight * (0 if cls in truth else 1)
            bucket["weight_total"] += weight
            bucket["audited"] += 1
            bucket["errors"] += 0 if cls in truth else 1
    report = {}
    for cls, bucket in per_class.items():
        report[cls] = {
            "ipw_false_release_rate": (
                round(bucket["weighted"] / bucket["weight_total"], 4)
                if bucket["weight_total"] else None
            ),
            "audited_releases": bucket["audited"],
            "observed_errors": bucket["errors"],
        }
    report["_meta"] = {
        "estimator": "Horvitz-Thompson inverse probability weighting",
        "suspect_queue_only_audits_excluded": suspect_audits,
        "note": "未收到抽核意见的发布不计为判对；无抽核命中时错误率不可估计",
    }
    return report


def evaluation_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """总体补充指标（不能替代按类/按渠道口径）。"""
    exact = int((y_pred == y_true).all(axis=1).sum())
    total = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = round(tp / (tp + fp), 4) if tp + fp else None
    recall = round(tp / (tp + fn), 4) if tp + fn else None
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else None
    return {
        "exact_match_rate": round(exact / total, 4) if total else None,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "n_labeled_records": total,
    }
