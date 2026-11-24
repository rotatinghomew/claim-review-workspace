"""FastAPI 入口：与 CLI 共用同一套服务函数，操作均可交叉核对。

启动：python -m uvicorn claimdesk.api:app --reload
所有写操作落在 local-state，原始 data/ 只读。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, ingest, lifecycle, models, pipeline, policy, store
from .labels import CLASSES

app = FastAPI(
    title="包装声明分类与栏目采用工作台",
    version="1.0",
    description="声明类型判读服务，不产出法规符合性结论",
)


@app.on_event("startup")
def _startup():
    config.ensure_dirs()


@app.get("/health")
def health():
    return {"status": "ok", "classes": list(CLASSES)}


class ImportRequest(BaseModel):
    file: str = Field(description="待导入 JSONL 的绝对路径")
    batch: str


@app.post("/imports")
def create_import(req: ImportRequest):
    try:
        return ingest.import_products(Path(req.file), req.batch)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.get("/imports/{batch_id}")
def get_import(batch_id: str):
    return {"batch_id": batch_id, "products": ingest.staged_products(batch_id)}


class TrainRequest(BaseModel):
    candidate: str = Field(pattern="^(lr-text|lr-rich)$")
    tag: Optional[str] = None
    moment: Optional[str] = None


@app.post("/models/train")
def train(req: TrainRequest):
    try:
        result = pipeline.train_candidate(req.candidate, moment=req.moment, tag=req.tag)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))
    return {"model_version": result["model_version"], "n_training": result["n_training"]}


@app.get("/models")
def list_all_models():
    return {"models": models.list_models()}


@app.post("/models/{model_version}/evaluate")
def evaluate(model_version: str):
    try:
        report = pipeline.evaluate_model(model_version)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return {
        name: {
            "overall": split["overall"],
            "per_class": split["per_class"],
            "per_channel": split["per_channel"],
            "chosen_thresholds": split["chosen_thresholds"],
            "tolerance_feasibility": {
                cls: {"feasible": f["feasible"]}
                for cls, f in split["tolerance_feasibility"].items()
            },
            "n_eval_unlabeled": split["n_eval_unlabeled"],
        }
        for name, split in report["splits"].items()
    }


@app.post("/models/compare")
def compare(payload: dict):
    versions = payload.get("models", [])
    try:
        return pipeline.compare_models(versions)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.post("/models/{model_version}/apply")
def apply(model_version: str, payload: Optional[dict] = None):
    moment = (payload or {}).get("moment")
    try:
        return pipeline.apply_model(model_version, moment=moment)["summary"]
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))


@app.get("/models/{model_version}/decisions/{product_id}")
def explain(model_version: str, product_id: str):
    try:
        return ingest.explain_decision(model_version, product_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/models/{model_version}/decisions")
def list_decisions(model_version: str, action: Optional[str] = None,
                   class_name: Optional[str] = None):
    rows = store.read_table(store.decision_path(model_version))
    if action:
        rows = [r for r in rows if r["action"] == action]
    if class_name:
        rows = [r for r in rows if r["class_name"] == class_name]
    return {"model_version": model_version, "count": len(rows), "decisions": rows}


class QueueRequest(BaseModel):
    model_version: str
    week_id: str
    audit_share: float = config.DEFAULT_AUDIT_SHARE
    audit_probability: float = config.DEFAULT_AUDIT_PROBABILITY
    seed: int = config.DEFAULT_SEED


@app.post("/queues")
def create_queue(req: QueueRequest):
    decisions = store.read_table(store.decision_path(req.model_version))
    if not decisions:
        raise HTTPException(404, "模型尚未应用，无决定可排队")
    tables = store.load_raw_tables()
    return policy.build_weekly_queue(
        decisions,
        tables["editors"],
        {c["class_name"]: c for c in tables["cost_policies"]},
        req.week_id,
        audit_share=req.audit_share,
        audit_probability=req.audit_probability,
        seed=req.seed,
    )


@app.get("/queues/{week_id}/assignments/{editor_id}")
def my_assignments(week_id: str, editor_id: str):
    payload = store.load_json(store.queue_path(week_id))
    if payload is None:
        raise HTTPException(404, "周队列不存在")
    return {
        "editor_id": editor_id,
        "items": payload["assignments"].get(editor_id, []),
    }


class ReviewRequest(BaseModel):
    product_id: str
    editor_id: str
    classes: list[str]
    valid_specs: list[str]
    status: str = Field(pattern="^(temporary|final)$")
    selected_by: str = Field(pattern="^(random_audit|suspect_queue)$")
    selection_probability: Optional[float] = None
    moment: Optional[str] = None


@app.post("/reviews")
def create_review(req: ReviewRequest):
    tables = store.load_raw_tables()
    products_index = store.index_by_id(ingest.all_products())
    editors = store.index_by_id(tables["editors"])
    if req.editor_id not in editors:
        raise HTTPException(404, "编辑不存在")
    if req.product_id not in products_index:
        raise HTTPException(404, "规格不存在")
    try:
        return lifecycle.submit_review(
            product=products_index[req.product_id],
            editor=editors[req.editor_id],
            classes=req.classes,
            valid_specs=req.valid_specs,
            status=req.status,
            selected_by=req.selected_by,
            selection_probability=req.selection_probability,
            products_index=products_index,
            moment=req.moment,
        )
    except (PermissionError, ValueError) as exc:
        raise HTTPException(403 if isinstance(exc, PermissionError) else 400, str(exc))


@app.post("/reviews/{review_id}/withdraw")
def withdraw(review_id: str, payload: Optional[dict] = None):
    moment = (payload or {}).get("moment")
    try:
        lifecycle.withdraw_review(review_id, moment=moment)
        return lifecycle.propagate_withdrawal(review_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class PublishRequest(BaseModel):
    model_version: str
    batch_id: Optional[str] = None


@app.post("/publications")
def publish(req: PublishRequest):
    tables = store.load_raw_tables()
    products_index = store.index_by_id(ingest.all_products())
    decisions = store.read_table(store.decision_path(req.model_version))
    if not decisions:
        raise HTTPException(404, "模型尚未应用")
    proposals = lifecycle.propose_adoptions(
        decisions, lifecycle.all_reviews(), products_index, req.model_version
    )
    return lifecycle.publish(proposals, req.model_version, batch_id=req.batch_id)


@app.post("/publications/rollback/{snapshot_id}")
def rollback(snapshot_id: str):
    try:
        return lifecycle.rollback(snapshot_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/publications")
def list_publications(status: Optional[str] = None):
    rows = store.read_table(store.adoption_path())
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return {"count": len(rows), "adoptions": rows}


@app.post("/models/diff")
def model_diff(payload: dict):
    try:
        current = store.read_table(store.decision_path(payload["current"]))
        candidate = store.read_table(store.decision_path(payload["candidate"]))
    except KeyError:
        raise HTTPException(404, "缺少模型版本决定文件")
    products_index = store.index_by_id(ingest.all_products())
    return lifecycle.diff_model_versions(current, candidate, products_index)


@app.get("/reconcile")
def reconcile(model_version: Optional[str] = None):
    return {
        "publication_records": lifecycle.reconcile_publication_records(),
        "model_state": pipeline.cross_check_state(model_version) if model_version else None,
    }
