"""人审、撤销与栏目采用的生命周期管理。

关键规则：
- 编辑提交按权限校验；无定稿权限只能产生 temporary 意见，永远不进训练真值；
- 定稿可以只适用于部分规格（valid_specs），共享印刷组的规格可共享一份意见，
  但销量影响按规格去重、分版后的规格不能继续借旧结论；
- 定稿被撤销时，已发布信息按“失去证据的范围”标注并停止自动采用；
  仍有独立定稿支撑的规格保持可用；
- 撤销意见不会被后续重训带回（labels 层统一过滤）；
- 发布前写快照，支持整批回退。
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Optional

from . import config, store
from .labels import (
    applicability,
    independent_support,
    is_active_at,
    is_final,
    spec_listed,
)


# ---------------------------------------------------------------- editorial

def next_review_id(existing: list[dict]) -> str:
    nums = [int(r["id"][1:]) for r in existing if r["id"].startswith("l") and r["id"][1:].isdigit()]
    return f"l{max(nums, default=-1) + 1}"


def submit_review(
    *,
    product: dict,
    editor: dict,
    classes: list[str],
    valid_specs: list[str],
    status: str,
    selected_by: str,
    selection_probability: Optional[float],
    products_index: dict[str, dict],
    moment: Optional[str] = None,
) -> dict:
    """编辑提交意见，含权限与适用规格校验。返回写入的记录。"""
    if not classes:
        raise ValueError("至少需要一个声明类型")
    permitted = set(editor.get("classes", []))
    blocked = [c for c in classes if c not in permitted]
    if blocked:
        raise PermissionError(f"编辑{editor['id']}无权定稿声明类: {blocked}")
    if status == "final" and not editor.get("can_finalize"):
        raise PermissionError(f"编辑{editor['id']}无定稿权限，只能提交临时意见")
    if status not in {"temporary", "final"}:
        raise ValueError("status 仅支持 temporary/final")
    for spec in valid_specs:
        target = products_index.get(spec)
        if target is None:
            raise ValueError(f"适用规格不存在: {spec}")
        if target.get("print_group") != product.get("print_group"):
            raise ValueError(f"规格{spec}包装印刷组不一致，不能跨文字套用意见")
    if product["id"] not in valid_specs:
        raise ValueError("valid_specs 必须包含被审规格本身")
    raw_reviews = store.read_table(config.DATA_DIR / "reviews.jsonl")
    submissions = store.read_table(store.editorial_path())
    review_id = next_review_id(raw_reviews + submissions)
    record = {
        "id": review_id,
        "product_id": product["id"],
        "reviewer": editor["id"],
        "classes": list(classes),
        "status": status,
        "selected_by": selected_by,
        "selection_probability": selection_probability,
        "valid_specs": list(valid_specs),
        "known_at": moment or store.now_iso(),
        "withdrawn_at": None,
    }
    store.append_row(store.editorial_path(), record)
    return record


def all_reviews() -> list[dict]:
    """原始意见 + 工作台提交，工作台记录覆盖同 id 时以工作台为准。"""
    merged = {r["id"]: r for r in store.read_table(config.DATA_DIR / "reviews.jsonl")}
    for row in store.read_table(store.editorial_path()):
        merged[row["id"]] = row
    return list(merged.values())


def withdraw_review(review_id: str, moment: Optional[str] = None) -> dict:
    """撤销意见：写入撤销时间，返回受影响采用范围。"""
    reviews = all_reviews()
    target = next((r for r in reviews if r["id"] == review_id), None)
    if target is None:
        raise KeyError(f"意见不存在: {review_id}")
    if target.get("withdrawn_at"):
        raise ValueError(f"意见{review_id}已被撤销")
    target = copy.deepcopy(target)
    target["withdrawn_at"] = moment or store.now_iso()
    # 工作台文件始终追加一条带撤销时间的覆盖记录
    store.append_row(store.editorial_path(), target)
    return target


# ---------------------------------------------------------------- adoptions

def _active_adoptions() -> list[dict]:
    return [a for a in store.read_table(store.adoption_path()) if a.get("status") == "active"]


def evidence_status(
    product_id: str,
    class_name: str,
    reviews: list[dict],
    products_index: dict[str, dict],
) -> dict:
    """当前时点某规格/声明类的证据状态。"""
    product = products_index[product_id]
    supports = [
        rv for rv in reviews
        if is_active_at(rv)
        and class_name in rv.get("classes", [])
        and applicability(rv, product, products_index)["applicable_for_adoption"]
    ]
    if supports:
        return {"status": "supported", "review_ids": [rv["id"] for rv in supports]}
    return {"status": "evidence_lost", "review_ids": []}


def propose_adoptions(
    decisions: list[dict],
    reviews: list[dict],
    products_index: dict[str, dict],
    model_version: str,
) -> list[dict]:
    """把 release 决定转成候选采用，逐条做线上资格检查。

    自动发布仍要求存在可支撑的正式定稿（抽核/历史定稿），
    模型输出本身不构成证据，也不会因为来自模型而自我增强可信度。
    """
    proposals = []
    for row in decisions:
        if row["action"] != "release":
            continue
        evidence = evidence_status(
            row["product_id"], row["class_name"], reviews, products_index
        )
        proposals.append({
            "product_id": row["product_id"],
            "class_name": row["class_name"],
            "model_version": model_version,
            "score": row["score"],
            "evidence": evidence["status"],
            "supporting_review_ids": evidence["review_ids"],
            "reasons": row["reasons"],
        })
    return proposals


def _dedupe_units(product_ids: list[str], products_index: dict[str, dict]) -> int:
    """共享印刷文字的规格可共享意见，但销量影响按规格去重，不重复计算。"""
    return sum(int(products_index[pid]["units_sold"]) for pid in set(product_ids))


def publish(
    proposals: list[dict],
    model_version: str,
    *,
    batch_id: Optional[str] = None,
) -> dict:
    """确认发布：写快照后落采用记录，仅 evidence=supported 的项目可发布。"""
    existing = _active_adoptions()
    snapshot_id = batch_id or f"snap-{store.now_iso().replace(':', '').replace('-', '')}"
    store.save_json(
        config.SNAPSHOT_DIR / f"{snapshot_id}.json",
        {"adoptions_before": existing, "model_version": model_version},
    )
    accepted, blocked = [], []
    known = {(a["product_id"], a["column"]) for a in existing}
    for proposal in proposals:
        key = (proposal["product_id"], proposal["class_name"])
        if key in known:
            blocked.append({**proposal, "block_reason": "栏目项已存在"})
            continue
        if proposal["evidence"] != "supported":
            blocked.append({**proposal, "block_reason": "缺少合格正式定稿支撑"})
            continue
        record = {
            "id": f"adopt-{model_version}-{proposal['product_id']}-{proposal['class_name']}",
            "product_id": proposal["product_id"],
            "column": proposal["class_name"],
            "review_id": proposal["supporting_review_ids"][0],
            "all_supporting_review_ids": proposal["supporting_review_ids"],
            "model_version": model_version,
            "adopted_at": store.now_iso(),
            "status": "active",
            "snapshot_id": snapshot_id,
        }
        accepted.append(record)
        known.add(key)
    for record in accepted:
        store.append_row(store.adoption_path(), record)
    return {
        "snapshot_id": snapshot_id,
        "published": accepted,
        "blocked": blocked,
        "published_units_deduped": _dedupe_units(
            [r["product_id"] for r in accepted], _products_index_safe()
        ),
    }


def _products_index_safe() -> dict[str, dict]:
    tables = store.load_raw_tables()
    return store.index_by_id(tables["products"])


def rollback(snapshot_id: str) -> dict:
    """按快照整批回退：移除该快照写入且仍为 active 的采用。"""
    snapshot = store.load_json(config.SNAPSHOT_DIR / f"{snapshot_id}.json")
    if snapshot is None:
        raise KeyError(f"快照不存在: {snapshot_id}")
    current = store.read_table(store.adoption_path())
    before_keys = {(r["product_id"], r["column"]) for r in snapshot["adoptions_before"]}
    changed = []
    for record in current:
        key = (record["product_id"], record["column"])
        if record.get("snapshot_id") == snapshot_id and key not in before_keys:
            record = {**record, "status": "rolled_back", "rolled_back_at": store.now_iso()}
            changed.append(record)
    store.write_table(
        store.adoption_path(),
        [r for r in current if r.get("snapshot_id") != snapshot_id] + changed,
    )
    return {"snapshot_id": snapshot_id, "rolled_back": changed}


def propagate_withdrawal(review_id: str) -> dict:
    """撤销定稿后，标注失去证据的栏目范围；有独立定稿的规格保持可用。"""
    tables = store.load_raw_tables()
    products_index = store.index_by_id(tables["products"])
    reviews = all_reviews()
    withdrawn = next(r for r in reviews if r["id"] == review_id)
    affected, retained = [], []
    for adoption in _active_adoptions():
        if review_id not in adoption.get("all_supporting_review_ids", [adoption.get("review_id")]):
            continue
        independent = independent_support(
            adoption["product_id"],
            adoption["column"],
            reviews,
            products_index,
            exclude_review=review_id,
        )
        if independent:
            retained.append({
                "adoption_id": adoption["id"],
                "product_id": adoption["product_id"],
                "column": adoption["column"],
                "remaining_review_ids": independent,
            })
        else:
            affected.append({
                "adoption_id": adoption["id"],
                "product_id": adoption["product_id"],
                "column": adoption["column"],
                "scope_specs": list(withdrawn.get("valid_specs", [])),
            })
    if affected:
        current = store.read_table(store.adoption_path())
        flagged_keys = {(a["product_id"], a["column"]) for a in affected}
        for record in current:
            if (record["product_id"], record["column"]) in flagged_keys and record.get("status") == "active":
                record["status"] = "evidence_lost"
                record["evidence_lost_at"] = store.now_iso()
                record["evidence_lost_review_id"] = review_id
        store.write_table(store.adoption_path(), current)
    return {
        "withdrawn_review_id": review_id,
        "evidence_lost_adoptions": affected,
        "retained_adoptions": retained,
        "note": "无关联采用时两个列表均为空，撤销本身仍已记录",
    }


# ---------------------------------------------------------------- model diff

def diff_model_versions(
    current_decisions: list[dict],
    candidate_decisions: list[dict],
    products_index: dict[str, dict],
) -> dict:
    """比较候选模型相对现行版本新增/撤下哪些栏目项。

    键为 (product_id, class_name)，只比较 release 动作；销量按规格去重。
    """
    def release_map(rows):
        return {(r["product_id"], r["class_name"]): r for r in rows if r["action"] == "release"}

    current = release_map(current_decisions)
    candidate = release_map(candidate_decisions)
    added = [candidate[k] for k in sorted(candidate.keys() - current.keys())]
    removed = [current[k] for k in sorted(current.keys() - candidate.keys())]
    kept = sorted(current.keys() & candidate.keys())
    return {
        "added": [{"product_id": r["product_id"], "class_name": r["class_name"],
                   "score": r["score"]} for r in added],
        "removed": [{"product_id": r["product_id"], "class_name": r["class_name"],
                     "score": r["score"]} for r in removed],
        "kept_count": len(kept),
        "added_units_deduped": _dedupe_units([r["product_id"] for r in added], products_index),
        "removed_units_deduped": _dedupe_units([r["product_id"] for r in removed], products_index),
    }


def reconcile_publication_records() -> dict:
    """核对既有 publication_records 的证据现状（含临时意见、撤销意见案例）。"""
    tables = store.load_raw_tables()
    products_index = store.index_by_id(tables["products"])
    reviews = {r["id"]: r for r in all_reviews()}
    findings = []
    for pub in tables["publication_records"]:
        review = reviews.get(pub["review_id"])
        if review is None:
            findings.append({**pub, "finding": "引用意见缺失"})
            continue
        if not is_final(review):
            findings.append({**pub, "finding": "引用的是临时意见，不能作为正式证据"})
        elif not is_active_at(review):
            findings.append({**pub, "finding": "引用定稿已撤销，栏目项失去证据"})
        elif not spec_listed(review, pub["product_id"]):
            findings.append({**pub, "finding": "规格不在意见适用范围内"})
        else:
            check = applicability(review, products_index[pub["product_id"]], products_index)
            if not check["applicable_for_adoption"]:
                findings.append({**pub, "finding": "规格已分版，旧结论不可继续采用"})
    return {"checked": len(tables["publication_records"]), "findings": findings}
