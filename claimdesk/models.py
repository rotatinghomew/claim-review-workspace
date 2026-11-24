"""可训练、可比较、可应用的多标签声明分类模型。

候选模型（开发者选定的两个可比较方案，均为线性模型，便于向编辑解释）：
- lr-text：仅包装文字 TF-IDF + 一对多逻辑回归；
- lr-rich：文字 + 配料 + 营养结构化标志的同一模型族。

多标签用 OneVsRest：每个声明类独立判 0/1，一条包装可同时命中多类。
模型输出是“建议概率”，既不是真值也不是合规结论；阈值由 policy 在
评估集上按业务容限选择，不写死在模型里。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from . import config, store
from .features import PackagingFeatures
from .labels import CLASSES

CANDIDATES = {
    "lr-text": {"feature_profile": "text_only", "class_weight": "balanced"},
    "lr-rich": {"feature_profile": "rich", "class_weight": "balanced"},
}


class ClaimClassifier:
    """特征 + 一对多逻辑回归的完整流水线。"""

    def __init__(self, candidate: str, classes=CLASSES):
        if candidate not in CANDIDATES:
            raise ValueError(f"未知候选模型: {candidate}")
        self.candidate = candidate
        self.classes = tuple(classes)
        spec = CANDIDATES[candidate]
        self.pipeline = Pipeline([
            ("features", PackagingFeatures(profile=spec["feature_profile"])),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=2000,
                        class_weight=spec["class_weight"],
                    )
                ),
            ),
        ])
        self.fitted = False
        self.metadata: dict = {}

    def fit(self, records: list[dict], y: np.ndarray) -> "ClaimClassifier":
        if not records:
            raise ValueError("训练集为空")
        if len(records) != len(y):
            raise ValueError("记录数与标签矩阵行数不一致")
        self.pipeline.fit(records, np.asarray(y))
        self.fitted = True
        return self

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("模型尚未训练")
        proba = self.pipeline.predict_proba(records)
        return np.asarray(proba, dtype=float)

    def predict(self, records: list[dict], thresholds: Optional[np.ndarray] = None) -> np.ndarray:
        proba = self.predict_proba(records)
        if thresholds is None:
            thresholds = np.full(len(self.classes), 0.5)
        return (proba >= np.asarray(thresholds)).astype(int)

    def save(self, model_version: str, metadata: dict) -> Path:
        if not self.fitted:
            raise RuntimeError("未训练模型不能保存")
        config.ensure_dirs()
        model_dir = config.MODELS_DIR / model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_dir / "pipeline.joblib")
        payload = {
            "model_version": model_version,
            "candidate": self.candidate,
            "classes": list(self.classes),
            "created_at": store.now_iso(),
            **metadata,
        }
        store.save_json(model_dir / "metadata.json", payload)
        return model_dir


def load_model(model_version: str) -> ClaimClassifier:
    path = config.MODELS_DIR / model_version / "pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"模型版本不存在: {model_version}")
    model = joblib.load(path)
    if not isinstance(model, ClaimClassifier):
        raise TypeError("模型文件内容不是 ClaimClassifier")
    return model


def model_metadata(model_version: str) -> dict:
    return store.load_json(config.MODELS_DIR / model_version / "metadata.json", {})


def list_models() -> list[dict]:
    if not config.MODELS_DIR.exists():
        return []
    out = []
    for model_dir in sorted(config.MODELS_DIR.iterdir()):
        meta = store.load_json(model_dir / "metadata.json", None)
        if meta:
            out.append(meta)
    return out


def make_version(candidate: str, tag: Optional[str] = None) -> str:
    stamp = time.strftime("%Y%m%d%H%M%S")
    suffix = f"-{tag}" if tag else ""
    return f"{candidate}-{stamp}{suffix}"
