"""自动发布 / 排队复核 / 暂缺资料三类决定，以及复核队列与抽核机制。

决定语义（按 声明类 × 规格 计算，再汇总到包装）：
- release：概率达到该类发布阈值，且该类容限在评估集上可守住；
- review ：介于复核阈与发布阈之间，或命中高影响/高分疑点，进入编辑队列；
- hold   ：资料不足（如总糖缺项且声明依赖数值旁证），暂缺资料，不自动判；
一条包装可以同时持有多个声明类的不同决定。

队列纪律：
- 高分疑点（suspect）解决当前不能发布的高影响对象；
- 随机抽核（random_audit）保留度量自动发布误差所必需的机会，
  按固定概率抽样并把“实际选中方式、入选概率、种子”逐条落盘；
- 编辑容量按周、按权限、按复核分钟分配；
- 抽核返回的意见不代表全目录错误率；未收到意见不等于判对。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config, store
from .labels import CLASSES


@dataclass
class Decision:
    product_id: str
    class_name: str
    action: str  # release / review / hold
    score: float
    threshold: float
    reasons: list[str]
    cost_weight: float


def choose_thresholds(
    sweep: dict[str, list[dict]],
    cost_policies: dict[str, dict],
    *,
    require_feasible: bool = True,
) -> tuple[dict[str, float], dict[str, dict]]:
    """在误放容限内选覆盖最大的阈值；不可行时退到最保守阈值并标记。

    每个声明类独立选阈。若所有阈值都守不住 maximum_error，
    选误放率最低的阈值并在 feasibility 中记录 breached，由发布流程拒绝自动放行该类。
    """
    thresholds: dict[str, float] = {}
    feasibility: dict[str, dict] = {}
    for cls in CLASSES:
        policy = cost_policies.get(cls, {"maximum_error": 0.05})
        tolerance = policy["maximum_error"]
        rows = sweep[cls]
        eligible = [r for r in rows if r["false_release_rate"] <= tolerance]
        if eligible:
            best = max(eligible, key=lambda r: (r["auto_coverage"], -r["threshold"]))
            thresholds[cls] = best["threshold"]
            feasibility[cls] = {
                "feasible": True,
                "tolerance": tolerance,
                "chosen": best,
            }
        else:
            safest = min(rows, key=lambda r: (r["false_release_rate"], -r["threshold"]))
            thresholds[cls] = safest["threshold"] if require_feasible else 0.95
            feasibility[cls] = {
                "feasible": False,
                "tolerance": tolerance,
                "best_effort": safest,
                "note": "评估集上无阈值能守住误放容限，该类不允许自动发布",
            }
    return thresholds, feasibility


def _sugar_dependent_gap(product: dict, class_name: str) -> bool:
    """资料缺口规则：总糖缺项时，依赖数值旁证的声称只能暂缺资料。

    文字类声明（未添加、口味宣传）不依赖总糖数值；总糖含量声称与
    无法判断类在 sugar_g 缺失时不得凭文字自动发布。
    """
    if product.get("sugar_g") is not None:
        return False
    return class_name in {"总糖含量声称", "无法判断"}


def decide_records(
    products: list[dict],
    proba: np.ndarray,
    thresholds: dict[str, float],
    feasibility: dict[str, dict],
    cost_policies: dict[str, dict],
    *,
    review_margin: float = 0.15,
) -> list[dict]:
    """生成逐 规格×声明类 的三类决定与可展示依据。"""
    decisions: list[dict] = []
    for i, product in enumerate(products):
        for j, cls in enumerate(CLASSES):
            score = float(proba[i, j])
            threshold = float(thresholds[cls])
            reasons: list[str] = []
            gap = _sugar_dependent_gap(product, cls)
            if gap:
                action = "hold"
                reasons.append("总糖数据缺项，该声明类需要数值旁证，暂缺资料")
            elif not feasibility.get(cls, {}).get("feasible", False):
                action = "review"
                reasons.append("评估集上该类误放容限不可守，禁止自动发布")
            elif score >= threshold:
                action = "release"
                reasons.append(f"模型概率{score:.3f}达到发布阈值{threshold:.3f}")
            elif score >= max(0.0, threshold - review_margin):
                action = "review"
                reasons.append(
                    f"概率{score:.3f}位于复核带[{max(0.0, threshold - review_margin):.3f},{threshold:.3f})"
                )
            else:
                action = "hold"
                reasons.append(f"概率{score:.3f}低于复核带下限，不产生该栏目项")
            decisions.append({
                "product_id": product["id"],
                "family": product["family"],
                "channel": product["channel"],
                "class_name": cls,
                "action": action,
                "score": round(score, 4),
                "threshold": round(threshold, 4),
                "reasons": reasons,
                "false_release_cost": cost_policies.get(cls, {}).get("false_release_cost"),
                "units_sold": product["units_sold"],
            })
    return decisions


def summarize_decisions(decisions: list[dict]) -> dict:
    """按声明类汇总三类决定的数量与销量影响（销量按规格去重由调用方处理）。"""
    summary: dict[str, dict] = {}
    for cls in CLASSES:
        rows = [d for d in decisions if d["class_name"] == cls]
        counts = {"release": 0, "review": 0, "hold": 0}
        units = {"release": 0, "review": 0, "hold": 0}
        for row in rows:
            counts[row["action"]] += 1
            units[row["action"]] += int(row["units_sold"])
        summary[cls] = {"counts": counts, "units_sold_touched": units}
    return summary


def build_weekly_queue(
    decisions: list[dict],
    editors: list[dict],
    cost_policies: dict[str, dict],
    week_id: str,
    *,
    audit_share: float = config.DEFAULT_AUDIT_SHARE,
    audit_probability: float = config.DEFAULT_AUDIT_PROBABILITY,
    seed: int = config.DEFAULT_SEED,
) -> dict:
    """按编辑周容量构造复核队列，并保留随机抽核机会。

    输出：
    - suspect_items：高分疑点，按 误放代价×销量×(1-概率) 排序；
    - audit_items：从自动发布集合按 audit_probability 抽出的抽核机会；
    - assignments：按编辑权限（可定稿声明类、周分钟）分配；
    - 每条抽核记录实际选中方式与入选概率，落 audit 日志。
    """
    rng = np.random.default_rng(seed)
    review_rows = [d for d in decisions if d["action"] == "review"]
    release_rows = [d for d in decisions if d["action"] == "release"]

    def priority(row):
        cost = float(row["false_release_cost"] or 1)
        return cost * int(row["units_sold"]) * (1.0 - float(row["score"]))

    suspect = sorted(review_rows, key=priority, reverse=True)

    audit_pool_flags = rng.random(len(release_rows))
    audit_items = []
    for row, draw in zip(release_rows, audit_pool_flags):
        selected = draw < audit_probability
        audit_items.append({
            **row,
            "draw": round(float(draw), 6),
            "selection_probability": audit_probability,
            "selected": bool(selected),
            "selection_method": "bernoulli_p_%.2f_seed_%d" % (audit_probability, seed),
        })
    selected_audit = [a for a in audit_items if a["selected"]]

    total_minutes = sum(int(e["weekly_minutes"]) for e in editors)
    audit_budget_minutes = math.floor(total_minutes * audit_share)
    suspect_budget_minutes = total_minutes - audit_budget_minutes

    def minutes_for(cls):
        return int(cost_policies.get(cls, {}).get("review_minutes", 5))

    chosen_audit, used = [], 0
    for item in selected_audit:
        need = minutes_for(item["class_name"])
        if used + need <= audit_budget_minutes:
            chosen_audit.append({**item, "stream": "random_audit"})
            used += need
    chosen_suspect, used_s = [], 0
    for item in suspect:
        need = minutes_for(item["class_name"])
        if used_s + need <= suspect_budget_minutes:
            chosen_suspect.append({**item, "stream": "suspect_queue"})
            used_s += need

    assignments = _assign(chosen_suspect + chosen_audit, editors, minutes_for)

    payload = {
        "week_id": week_id,
        "generated_at": store.now_iso(),
        "capacity": {
            "total_editor_minutes": total_minutes,
            "audit_budget_minutes": audit_budget_minutes,
            "suspect_budget_minutes": suspect_budget_minutes,
            "audit_minutes_used": used,
            "suspect_minutes_used": used_s,
        },
        "suspect_items": chosen_suspect,
        "audit_items": chosen_audit,
        "audit_pool_full": audit_items,
        "assignments": assignments,
        "deferred_suspect_count": max(0, len(suspect) - len(chosen_suspect)),
        "audit_opportunities_selected": len(chosen_audit),
        "audit_opportunities_available": len(selected_audit),
    }
    store.save_json(store.queue_path(week_id), payload)
    for item in chosen_audit:
        store.append_row(store.audit_log_path(), {
            "week_id": week_id,
            "product_id": item["product_id"],
            "class_name": item["class_name"],
            "selection_method": item["selection_method"],
            "selection_probability": item["selection_probability"],
            "draw": item["draw"],
            "recorded_at": store.now_iso(),
        })
    store.write_table(
        store.assignment_path(week_id),
        [
            {"week_id": week_id, "editor_id": editor_id, **item}
            for editor_id, items in assignments.items()
            for item in items
        ],
    )
    return payload


def _assign(items: list[dict], editors: list[dict], minutes_for) -> dict[str, list[dict]]:
    """按编辑可定稿声明类与剩余分钟分配；can_finalize=False 的编辑只领临时复核。"""
    load = {e["id"]: int(e["weekly_minutes"]) for e in editors}
    perms = {e["id"]: set(e["classes"]) for e in editors}
    can_final = {e["id"]: bool(e["can_finalize"]) for e in editors}
    assignments: dict[str, list[dict]] = {e["id"]: [] for e in editors}
    for item in items:
        cls = item["class_name"]
        candidates = [
            eid for eid in load
            if cls in perms[eid] and load[eid] >= minutes_for(cls)
        ]
        if not candidates:
            continue
        # 优先分给可定稿编辑，再按剩余时间最充裕者均衡
        candidates.sort(key=lambda eid: (not can_final[eid], -load[eid]))
        editor_id = candidates[0]
        load[editor_id] -= minutes_for(cls)
        assignments[editor_id].append({
            "product_id": item["product_id"],
            "class_name": cls,
            "stream": item["stream"],
            "allocated_minutes": minutes_for(cls),
            "requires_finalizer": not can_final[editor_id],
        })
    return {k: v for k, v in assignments.items() if v}
