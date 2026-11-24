"""路径与运行参数。所有可写状态集中在 local-state，原始 data/ 只读。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "local-state"

MODELS_DIR = STATE_DIR / "models"
EVAL_DIR = STATE_DIR / "evaluations"
QUEUE_DIR = STATE_DIR / "queues"
EDIT_DIR = STATE_DIR / "editorial"
ADOPT_DIR = STATE_DIR / "adoptions"
AUDIT_DIR = STATE_DIR / "audit"
SNAPSHOT_DIR = STATE_DIR / "snapshots"
IMPORTS_DIR = STATE_DIR / "imports"

WRITABLE_DIRS = (
    MODELS_DIR, EVAL_DIR, QUEUE_DIR, EDIT_DIR,
    ADOPT_DIR, AUDIT_DIR, SNAPSHOT_DIR, IMPORTS_DIR,
)

# 抽核：对自动发布对象按此概率保留随机抽核机会（合成资料中既有抽核为 0.15）
DEFAULT_AUDIT_PROBABILITY = 0.15
# 编辑工作量中预留给随机抽核的比例，其余用于高分疑点
DEFAULT_AUDIT_SHARE = 0.4
# 抽核随机种子，保证“实际选中方式”可复现
DEFAULT_SEED = 20261124

TABLE_NAMES = ("products", "reviews", "editors", "publication_records", "cost_policies")


def ensure_dirs() -> None:
    for directory in WRITABLE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def use_state_root(state_root: Path) -> None:
    """把全部可写状态重定向到独立目录（测试隔离用）。"""
    global STATE_DIR, MODELS_DIR, EVAL_DIR, QUEUE_DIR, EDIT_DIR
    global ADOPT_DIR, AUDIT_DIR, SNAPSHOT_DIR, IMPORTS_DIR
    STATE_DIR = Path(state_root)
    MODELS_DIR = STATE_DIR / "models"
    EVAL_DIR = STATE_DIR / "evaluations"
    QUEUE_DIR = STATE_DIR / "queues"
    EDIT_DIR = STATE_DIR / "editorial"
    ADOPT_DIR = STATE_DIR / "adoptions"
    AUDIT_DIR = STATE_DIR / "audit"
    SNAPSHOT_DIR = STATE_DIR / "snapshots"
    IMPORTS_DIR = STATE_DIR / "imports"
    ensure_dirs()
