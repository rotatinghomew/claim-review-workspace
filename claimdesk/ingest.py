"""新记录导入与依据查询。

导入的外来文件按“不信任数据”处理：单独落到 local-state/imports/<批次>/，
逐字段按 schema 校验、编号去重、跨表引用检查；合格记录进入工作台视图，
不覆盖 data/ 原始资料。
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config, store

REQUIRED_PRODUCT_FIELDS = (
    "id", "family", "version", "valid_from", "channel", "text",
    "ingredients", "basis", "basis_g", "sugar_g", "units_sold",
    "known_at", "print_group",
)


def import_products(path: str | Path, batch_id: str) -> dict:
    """校验并导入一批新的产品包装记录。"""
    source = Path(path)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    batch_dir = config.IMPORTS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    # 原样留档，保证可追溯
    (batch_dir / source.name).write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    tables = store.load_raw_tables()
    known_ids = {p["id"] for p in tables["products"]}
    imported = store.read_table(batch_dir / "products.jsonl")
    staged_ids = {p["id"] for p in imported}
    accepted, rejected = [], []
    for row in rows:
        problems = []
        missing = [f for f in REQUIRED_PRODUCT_FIELDS if f not in row]
        if missing:
            problems.append(f"缺字段:{missing}")
        if row.get("id") in known_ids:
            problems.append("编号与原始资料冲突")
        if row.get("id") in staged_ids:
            problems.append("批次内编号重复")
        if row.get("basis") not in {"perServing", "per100g"}:
            problems.append("basis 取值非法")
        if row.get("sugar_g") is not None and not isinstance(row.get("sugar_g"), (int, float)):
            problems.append("sugar_g 必须是数值或 null")
        if problems:
            rejected.append({"id": row.get("id"), "problems": problems})
        else:
            accepted.append(row)
            staged_ids.add(row["id"])
    store.write_table(batch_dir / "products.jsonl", imported + accepted)
    return {
        "batch_id": batch_id,
        "received": len(rows),
        "accepted": len(accepted),
        "rejected": rejected,
        "staged_path": str(batch_dir / "products.jsonl"),
    }


def staged_products(batch_id: str) -> list[dict]:
    return store.read_table(config.IMPORTS_DIR / batch_id / "products.jsonl")


def all_products() -> list[dict]:
    """原始产品 + 所有导入批次的合格记录（工作台视图，不改原始表）。"""
    tables = store.load_raw_tables()
    products = list(tables["products"])
    seen = {p["id"] for p in products}
    if config.IMPORTS_DIR.exists():
        for batch_dir in sorted(config.IMPORTS_DIR.iterdir()):
            for row in store.read_table(batch_dir / "products.jsonl"):
                if row["id"] not in seen:
                    products.append(row)
                    seen.add(row["id"])
    return products


def explain_decision(model_version: str, product_id: str) -> dict:
    """查看某规格发布或拒判的具体依据：模型概率、阈值、特征标志、证据状态。"""
    from .features import PackagingFeatures
    from . import lifecycle
    decisions = store.read_table(store.decision_path(model_version))
    rows = [d for d in decisions if d["product_id"] == product_id]
    if not rows:
        raise KeyError(f"该规格没有模型输出: {product_id}")
    products_index = store.index_by_id(all_products())
    product = products_index[product_id]
    feature_explain = PackagingFeatures().explain_flags(product)
    evidence = {}
    reviews = lifecycle.all_reviews()
    for row in rows:
        evidence[row["class_name"]] = lifecycle.evidence_status(
            product_id, row["class_name"], reviews, products_index
        )
    return {
        "product_id": product_id,
        "packaging_text": product["text"],
        "ingredients": product["ingredients"],
        "nutrition": {
            "basis": product["basis"],
            "basis_g": product["basis_g"],
            "sugar_g": product["sugar_g"],
        },
        "feature_flags": feature_explain,
        "decisions": [
            {
                "class_name": row["class_name"],
                "action": row["action"],
                "score": row["score"],
                "threshold": row["threshold"],
                "reasons": row["reasons"],
                "evidence": evidence[row["class_name"]],
            }
            for row in rows
        ],
        "disclaimer": "本依据是声明类型判读材料，不是法规符合性结论",
    }
