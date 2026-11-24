"""资料读取与本地状态持久化。

data/ 下的原始表只读；工作台产生的模型输出、人审状态、栏目采用状态、
队列与抽核记录全部写入 local-state，且互相通过稳定编号关联，可交叉核对。
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def now_iso() -> str:
    """统一的带时区时间戳。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_raw_tables(data_dir: Path | None = None) -> dict[str, list[dict]]:
    """读取五张原始表（只读，不做业务判决）。"""
    base = data_dir or config.DATA_DIR
    return {name: _read_jsonl(base / f"{name}.jsonl") for name in config.TABLE_NAMES}


def index_by_id(rows: Iterable[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(path: Path, rows: Iterable[dict]) -> None:
    _write_jsonl(path, rows)


def append_row(path: Path, row: dict) -> None:
    _append_jsonl(path, row)


def read_table(path: Path) -> list[dict]:
    return _read_jsonl(path)


def editorial_path() -> Path:
    return config.EDIT_DIR / "submissions.jsonl"


def decision_path(model_version: str) -> Path:
    return config.EDIT_DIR / f"decisions_{model_version}.jsonl"


def adoption_path() -> Path:
    return config.ADOPT_DIR / "adoptions.jsonl"


def queue_path(week_id: str) -> Path:
    return config.QUEUE_DIR / f"queue_{week_id}.jsonl"


def assignment_path(week_id: str) -> Path:
    return config.QUEUE_DIR / f"assignments_{week_id}.jsonl"


def audit_log_path() -> Path:
    return config.AUDIT_DIR / "selections.jsonl"