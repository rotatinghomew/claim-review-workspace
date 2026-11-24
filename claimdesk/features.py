"""包装特征：文字 n-gram + 显式判读标志 + 营养数值（含每份/每百克归一）。

设计约束：
- 营养表的“0”本身不构成无糖/未添加糖结论，数值只作为文字声明的旁证特征；
- sugar_g 缺失用独立缺失标志表达，不能插补成 0；
- perServing 与 per100g 先统一折算到每百克，避免把基准混用学成冲突；
- 销量不进特征（只用于影响面加权），渠道不进特征（渠道可靠性单独评估）。
"""
from __future__ import annotations

import re

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# 显式短语标志：名称 -> 正则。保留显式清单便于向编辑解释模型依据。
TEXT_FLAGS: dict[str, str] = {
    "flag_no_added_phrase": r"未添加|零添加|0蔗糖|无添加",
    "flag_total_sugar_phrase": r"总糖|每百克.*克|含糖量",
    "flag_taste_phrase": r"口味|口感|清爽|甜腻|好喝",
    "flag_missing_sugar_phrase": r"未列明|未标注|暂无.*糖|糖一栏",
    "flag_natural_sugar_phrase": r"天然含糖|乳糖|果糖|原料.*糖",
}

NUMERIC_FLAG_NAMES = (
    "sugar_missing",
    "sugar_per_100g",
    "basis_per_serving",
    "basis_per_100g",
    "ingredients_len",
)


class StructuredFeatures(BaseEstimator, TransformerMixin):
    """营养数值与显式标志组成的稠密特征块。"""

    def fit(self, records, y=None):
        return self

    @staticmethod
    def _one_record(record: dict) -> list[float]:
        missing = record.get("sugar_g") is None
        if missing:
            per_100g = 0.0
        elif record.get("basis") == "perServing" and record.get("basis_g"):
            per_100g = float(record["sugar_g"]) * 100.0 / float(record["basis_g"])
        else:
            per_100g = float(record["sugar_g"])
        per_100g = min(per_100g, 100.0) / 100.0  # 缩放到 0~1，避免量纲主导
        text = record.get("text", "") or ""
        flags = [1.0 if re.search(pattern, text) else 0.0
                 for pattern in TEXT_FLAGS.values()]
        structured = [
            1.0 if missing else 0.0,
            per_100g,
            1.0 if record.get("basis") == "perServing" else 0.0,
            1.0 if record.get("basis") == "per100g" else 0.0,
            min(len(record.get("ingredients", "") or ""), 200) / 200.0,
        ]
        return flags + structured

    def transform(self, records):
        rows = [self._one_record(dict(r)) for r in records]
        return csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.array(list(TEXT_FLAGS) + list(NUMERIC_FLAG_NAMES), dtype=object)


class PackagingFeatures(BaseEstimator, TransformerMixin):
    """组合特征：包装文字 TF-IDF + 配料 TF-IDF + 结构化标志。

    profile="text_only" 仅保留包装文字，用于与富特征候选做模型比较。
    """

    def __init__(self, profile: str = "rich"):
        self.profile = profile

    def fit(self, records, y=None):
        seq = list(records)
        self.text_vec_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(1, 3), min_df=1,
        )
        self.text_vec_.fit([r.get("text", "") or "" for r in seq])
        if self.profile == "rich":
            self.ing_vec_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
            self.ing_vec_.fit([r.get("ingredients", "") or "" for r in seq])
            self.struct_ = StructuredFeatures().fit(seq)
        return self

    def transform(self, records):
        seq = list(records)
        blocks = [self.text_vec_.transform([r.get("text", "") or "" for r in seq])]
        if self.profile == "rich":
            blocks.append(self.ing_vec_.transform([r.get("ingredients", "") or "" for r in seq]))
            blocks.append(self.struct_.transform(seq))
        return hstack(blocks, format="csr")

    def get_feature_names_out(self, input_features=None):
        names = [f"text::{n}" for n in self.text_vec_.get_feature_names_out()]
        if self.profile == "rich":
            names += [f"ingredients::{n}" for n in self.ing_vec_.get_feature_names_out()]
            names += [f"structured::{n}" for n in self.struct_.get_feature_names_out()]
        return np.array(names, dtype=object)

    def explain_flags(self, record: dict) -> dict:
        """返回命中的显式标志与营养折算，供“查看发布或拒判依据”展示。"""
        text = record.get("text", "") or ""
        hits = {
            name: bool(re.search(pattern, text))
            for name, pattern in TEXT_FLAGS.items()
        }
        missing = record.get("sugar_g") is None
        if missing:
            normalized = None
        elif record.get("basis") == "perServing" and record.get("basis_g"):
            normalized = round(float(record["sugar_g"]) * 100.0 / float(record["basis_g"]), 3)
        else:
            normalized = float(record["sugar_g"])
        return {
            "matched_flags": [k for k, v in hits.items() if v],
            "sugar_missing": missing,
            "sugar_per_100g_derived": normalized,
            "basis": record.get("basis"),
        }
