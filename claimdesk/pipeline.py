"""训练、比较与核验入口：把资格规则、模型、评估与阈值选择串成可复现流程。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, evaluate, lifecycle, models, policy, store
from .labels import CLASSES, label_matrix, parse_ts, training_labels


def _records_for(products, ids):
    index = {p["id"]: p for p in products}
    return [index[pid] for pid in ids if pid in index]


def train_candidate(
    candidate: str,
    *,
    moment: str | None = None,
    tag: str | None = None,
) -> dict:
    """只用合格正式定稿训练一个候选模型，并保存训练资格审计信息。"""
    tables = store.load_raw_tables()
    products = tables["products"]
    reviews = lifecycle.all_reviews()
    products_index = store.index_by_id(products)
    point = parse_ts(moment) if moment else None
    labels = training_labels(products, reviews, products_index, moment=point)

    eligible_ids = sorted(labels)
    records = _records_for(products, eligible_ids)
    y = np.asarray(label_matrix(records, labels), dtype=int)
    if y.sum() == 0:
        raise RuntimeError("没有任何合格训练标签")

    model = models.ClaimClassifier(candidate).fit(records, y)
    version = models.make_version(candidate, tag)
    model_dir = model.save(version, {
        "training": {
            "n_records": len(records),
            "labeled_positives_per_class": {
                cls: int(y[:, j].sum()) for j, cls in enumerate(CLASSES)
            },
            "excluded_temporary_reviews": sum(1 for r in reviews if r["status"] == "temporary"),
            "excluded_withdrawn_or_unknown_at_moment": sum(
                1 for r in reviews
                if r["status"] == "final" and not _active_for(r, point)
            ) + sum(
                1 for r in reviews
                if r["status"] == "final" and point is not None
                and not _known(r, point)
            ),
            "moment": moment,
            "unlabeled_are_unlabeled_not_negative": True,
        }
    })
    return {"model_version": version, "model_dir": str(model_dir),
            "n_training": len(records), "labels": labels}


def _active_for(review, point):
    from .labels import is_active_at
    return is_active_at(review, point)


def _known(review, point):
    from .labels import was_known_at
    return was_known_at(review, point)


def evaluate_model(model_version: str, seed: int = config.DEFAULT_SEED) -> dict:
    """两类外推评估 + 按类/按渠道可靠性 + 容限选阈，报告落盘。"""
    model = models.load_model(model_version)
    tables = store.load_raw_tables()
    products = tables["products"]
    reviews = lifecycle.all_reviews()
    products_index = store.index_by_id(products)
    labels = training_labels(products, reviews, products_index)
    cost_policies = {c["class_name"]: c for c in tables["cost_policies"]}

    splits = [
        evaluate.temporal_split(products),
        evaluate.family_holdout_split(products, seed=seed),
    ]
    report = {"model_version": model_version, "splits": {}}
    for split in splits:
        train_ids = evaluate.labeled_only(split.train_ids, labels)
        eval_ids = evaluate.labeled_only(split.eval_ids, labels)
        split_model = models.ClaimClassifier(model.candidate).fit(
            _records_for(products, train_ids),
            np.asarray(label_matrix(_records_for(products, train_ids), labels)),
        )
        eval_records = _records_for(products, eval_ids)
        y_true = np.asarray(label_matrix(eval_records, labels))
        proba = split_model.predict_proba(eval_records)
        y_pred = split_model.predict(eval_records)
        sweep = evaluate.threshold_sweep(proba, y_true)
        thresholds, feasibility = policy.choose_thresholds(sweep, cost_policies)
        report["splits"][split.name] = {
            "note": split.note,
            "n_train_labeled": len(train_ids),
            "n_eval_labeled": len(eval_ids),
            "n_eval_unlabeled": len(split.eval_ids) - len(eval_ids),
            "overall": evaluate.evaluation_summary(y_true, y_pred),
            "per_class": evaluate.per_class_reliability(y_true, y_pred),
            "per_channel": evaluate.per_channel_reliability(eval_records, y_true, y_pred),
            "threshold_sweep": sweep,
            "chosen_thresholds": thresholds,
            "tolerance_feasibility": feasibility,
        }
    path = config.EVAL_DIR / f"{model_version}.json"
    store.save_json(path, report)
    return report


def compare_models(versions: list[str]) -> dict:
    """并排比较候选模型在两个切分上的按类可靠性与容限可行性。"""
    comparison = {}
    for version in versions:
        report = store.load_json(config.EVAL_DIR / f"{version}.json")
        if report is None:
            raise FileNotFoundError(f"缺少评估报告: {version}，请先 evaluate")
        rows = {}
        for split_name, split in report["splits"].items():
            rows[split_name] = {
                cls: {
                    "false_release_rate": split["per_class"][cls]["false_release_rate"],
                    "miss_rate": split["per_class"][cls]["miss_rate"],
                    "auto_coverage": split["per_class"][cls]["auto_coverage"],
                    "tolerance_feasible": split["tolerance_feasibility"][cls]["feasible"],
                }
                for cls in CLASSES
            }
        comparison[version] = rows
    return comparison


def apply_model(model_version: str, *, moment: str | None = None) -> dict:
    """对当前全目录生成逐 规格×声明类 决定（三类），并保存模型输出。"""
    model = models.load_model(model_version)
    tables = store.load_raw_tables()
    products = tables["products"]
    cost_policies = {c["class_name"]: c for c in tables["cost_policies"]}
    eval_report = store.load_json(config.EVAL_DIR / f"{model_version}.json")
    if eval_report is None:
        raise RuntimeError("必须先完成评估并确定阈值，才能应用模型")
    # 阈值取未见家族切分（最保守外推口径）
    chosen = eval_report["splits"]["unseen_families"]["chosen_thresholds"]
    feasibility = eval_report["splits"]["unseen_families"]["tolerance_feasibility"]
    thresholds = {cls: float(chosen[cls]) for cls in CLASSES}
    proba = model.predict_proba(products)
    decisions = policy.decide_records(products, proba, thresholds, feasibility, cost_policies)
    store.write_table(store.decision_path(model_version), decisions)
    # 抽核视角的发布误差估计
    release_rows = [
        {"product_id": d["product_id"],
         "predicted_classes": [d["class_name"]] if d["action"] == "release" else []}
        for d in decisions
    ]
    release_rows = [r for r in release_rows if r["predicted_classes"]]
    ipw = evaluate.ipw_release_error(release_rows, lifecycle.all_reviews(),
                                     store.index_by_id(products))
    payload = {
        "model_version": model_version,
        "applied_at": store.now_iso(),
        "moment": moment,
        "decision_summary": policy.summarize_decisions(decisions),
        "ipw_release_error": ipw,
    }
    store.save_json(config.EDIT_DIR / f"apply_{model_version}.json", payload)
    return {"decisions": decisions, "summary": payload}


def cross_check_state(model_version: str) -> dict:
    """模型输出、人审状态、栏目采用状态三方核对。"""
    decisions = store.read_table(store.decision_path(model_version))
    reviews = lifecycle.all_reviews()
    adoptions = store.read_table(store.adoption_path())
    raw_tables = store.load_raw_tables()
    products_index = store.index_by_id(raw_tables["products"])
    decision_index = {(d["product_id"], d["class_name"]): d for d in decisions}
    issues = []
    for adoption in adoptions:
        key = (adoption["product_id"], adoption["column"])
        decision = decision_index.get(key)
        if decision is None:
            issues.append({"type": "adoption_without_model_output", "key": list(key)})
        elif adoption.get("status") == "active" and decision["action"] != "release":
            issues.append({"type": "active_adoption_not_released_by_model",
                           "key": list(key), "model_action": decision["action"]})
    # 人审资格核对：临时/撤销意见不应支撑 active 采用
    review_index = {r["id"]: r for r in reviews}
    for adoption in adoptions:
        if adoption.get("status") != "active":
            continue
        for rid in adoption.get("all_supporting_review_ids", [adoption.get("review_id")]):
            review = review_index.get(rid)
            if review is None or review["status"] != "final" or review.get("withdrawn_at"):
                issues.append({"type": "adoption_with_ineligible_review",
                               "adoption_id": adoption["id"], "review_id": rid})
    # 模型输出覆盖完整性
    covered = {d["product_id"] for d in decisions}
    missing = sorted({p["id"] for p in raw_tables["products"]} - covered)
    return {
        "model_version": model_version,
        "n_decisions": len(decisions),
        "n_adoptions": len(adoptions),
        "n_reviews": len(reviews),
        "products_missing_output": missing,
        "issues": issues,
        "status": "consistent" if not issues and not missing else "issues_found",
    }
