"""人审意见的资格规则：训练真值与线上采用分别检查。

核心原则（对应资料协议与运营要求）：
1. 临时意见（status=temporary）任何时候都不是真值，不能进训练，也不能支撑自动发布。
2. 被撤销意见在撤销时点之后失去证据效力；撤销时点之前的历史评估按“当时可知”解释。
3. 意见只适用于 valid_specs 明确列出、且包装印刷文字同组（print_group）的规格。
4. 规格后来分版（同家族出现更新版本）后，旧结论不能再借给当前栏目采用；
   但作为该历史包装文字的训练标签仍然有效。
5. 模型自身输出不作为意见资格依据，不能因为来自模型就增强可信度。
6. 没有合格意见的记录是“未标注”，不是负样本，也不是判对。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

CLASSES = ("未添加某种糖", "总糖含量声称", "无关口味宣传", "无法判断")


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def is_final(review: dict) -> bool:
    return review.get("status") == "final"


def is_active_at(review: dict, moment: Optional[datetime] = None) -> bool:
    """定稿且在 moment（含）之前未被撤销。moment=None 表示当前时点。"""
    if not is_final(review):
        return False
    withdrawn_at = parse_ts(review.get("withdrawn_at"))
    if withdrawn_at is None:
        return True
    if moment is None:
        return False
    return withdrawn_at > moment


def was_known_at(review: dict, moment: datetime) -> bool:
    known_at = parse_ts(review.get("known_at"))
    return known_at is not None and known_at <= moment


def spec_listed(review: dict, product_id: str) -> bool:
    return product_id in set(review.get("valid_specs", []))


def is_current_spec(product: dict, products_index: dict[str, dict]) -> bool:
    """该规格在其家族内是否仍为最新版本（未被分版取代）。"""
    family = product["family"]
    own_version = product["version"]
    for other in products_index.values():
        if other["family"] == family and other["version"] > own_version:
            return False
    return True


def applicability(review: dict, product: dict, products_index: dict[str, dict]) -> dict:
    """返回意见对某规格的适用检查明细，所有判定都给出可展示的原因。"""
    reviewed = products_index.get(review["product_id"])
    reasons: list[str] = []
    listed = spec_listed(review, product["id"])
    if not listed:
        reasons.append("规格不在意见的valid_specs范围内")
    print_match = (
        reviewed is not None
        and reviewed.get("print_group") == product.get("print_group")
    )
    if reviewed is not None and not print_match:
        reasons.append("包装印刷组与被审规格不一致，不能跨文字借用")
    superseded = not is_current_spec(product, products_index)
    if superseded:
        reasons.append("该规格所在家族已有更新版本，旧结论不能继续借用到当前栏目")
    return {
        "listed": listed,
        "print_match": print_match,
        "superseded": superseded,
        "applicable_for_training": listed and print_match,
        "applicable_for_adoption": listed and print_match and not superseded,
        "reasons": reasons,
    }


def eligible_reviews_for_product(
    product: dict,
    reviews: list[dict],
    products_index: dict[str, dict],
    *,
    moment: Optional[datetime] = None,
) -> list[dict]:
    """当前/指定时点对该规格有效的正式定稿（含适用性明细）。"""
    out = []
    for review in reviews:
        if not is_active_at(review, moment):
            continue
        if moment is not None and not was_known_at(review, moment):
            continue
        check = applicability(review, product, products_index)
        if check["applicable_for_adoption"]:
            out.append(review)
    return out


def training_labels(
    products: list[dict],
    reviews: list[dict],
    products_index: dict[str, dict],
    *,
    moment: Optional[datetime] = None,
) -> dict[str, set[str]]:
    """构造训练真值：product_id -> 声明类集合（支持一个包装多类声明）。

    只采用 moment 时点可知、未撤销的正式定稿；被审历史包装即使已分版，
    其文字层面的标签仍可用于训练。未命中任何合格定稿的产品不出现在结果中，
    调用方必须把“不在结果中”理解为未标注，而非负样本。
    """
    labels: dict[str, set[str]] = {}
    for review in reviews:
        if not is_active_at(review, moment):
            continue
        if moment is not None and not was_known_at(review, moment):
            continue
        for spec in review.get("valid_specs", []):
            product = products_index.get(spec)
            if product is None:
                continue
            check = applicability(review, product, products_index)
            if not check["applicable_for_training"]:
                continue
            labels.setdefault(spec, set()).update(review.get("classes", []))
    return labels


def label_matrix(
    products: list[dict], labels: dict[str, set[str]]
) -> list[list[int]]:
    """把多标签集合转成与 CLASSES 对齐的 0/1 矩阵；未标注产品被排除由调用方负责。"""
    return [[1 if cls in labels.get(p["id"], set()) else 0 for cls in CLASSES]
            for p in products]


def independent_support(
    product_id: str,
    class_name: str,
    reviews: list[dict],
    products_index: dict[str, dict],
    *,
    exclude_review: Optional[str] = None,
    moment: Optional[datetime] = None,
) -> list[str]:
    """除 exclude_review 外，仍能独立支撑该规格/声明类的定稿编号。

    撤销场景下用它判断“仍有独立定稿的规格保持可用”。
    """
    product = products_index.get(product_id)
    if product is None:
        return []
    support = []
    for review in eligible_reviews_for_product(
        product, reviews, products_index, moment=moment
    ):
        if review["id"] == exclude_review:
            continue
        if class_name in review.get("classes", []):
            support.append(review["id"])
    return support
