"""命令行入口：导入、训练、评估、比较、应用、队列、定稿、发布全流程。

示例：
  python -m claimdesk train --candidate lr-rich --tag v1
  python -m claimdesk evaluate --model lr-rich-...-v1
  python -m claimdesk compare --models a b
  python -m claimdesk apply --model lr-rich-...-v1
  python -m claimdesk queue --model lr-rich-...-v1 --week 2026-W48
  python -m claimdesk publish --model lr-rich-...-v1
"""
from __future__ import annotations

import argparse
import json

from . import config, ingest, lifecycle, pipeline, policy, store
from .features import PackagingFeatures


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def cmd_import(args) -> None:
    _print(ingest.import_products(args.file, args.batch))


def cmd_train(args) -> None:
    result = pipeline.train_candidate(args.candidate, moment=args.moment, tag=args.tag)
    _print({k: v for k, v in result.items() if k != "labels"})


def cmd_evaluate(args) -> None:
    report = pipeline.evaluate_model(args.model)
    summary = {
        "model_version": report["model_version"],
        "splits": {
            name: {
                "note": split["note"],
                "n_train_labeled": split["n_train_labeled"],
                "n_eval_labeled": split["n_eval_labeled"],
                "n_eval_unlabeled": split["n_eval_unlabeled"],
                "overall": split["overall"],
                "per_class": split["per_class"],
                "per_channel": split["per_channel"],
                "chosen_thresholds": split["chosen_thresholds"],
                "tolerance_feasibility": {
                    cls: {"feasible": f["feasible"]}
                    for cls, f in split["tolerance_feasibility"].items()
                },
            }
            for name, split in report["splits"].items()
        },
    }
    _print(summary)


def cmd_compare(args) -> None:
    _print(pipeline.compare_models(args.models))


def cmd_apply(args) -> None:
    result = pipeline.apply_model(args.model, moment=args.moment)
    _print(result["summary"])


def cmd_explain(args) -> None:
    _print(ingest.explain_decision(args.model, args.product))


def cmd_queue(args) -> None:
    decisions = store.read_table(store.decision_path(args.model))
    tables = store.load_raw_tables()
    payload = policy.build_weekly_queue(
        decisions,
        tables["editors"],
        {c["class_name"]: c for c in tables["cost_policies"]},
        args.week,
        audit_share=args.audit_share,
        audit_probability=args.audit_probability,
        seed=args.seed,
    )
    _print({
        "week_id": payload["week_id"],
        "capacity": payload["capacity"],
        "suspect_count": len(payload["suspect_items"]),
        "audit_selected": payload["audit_opportunities_selected"],
        "assignments": payload["assignments"],
        "deferred_suspect_count": payload["deferred_suspect_count"],
    })


def cmd_assignments(args) -> None:
    payload = store.load_json(config.queue_path(args.week))
    _print(payload["assignments"])


def cmd_submit(args) -> None:
    tables = store.load_raw_tables()
    products_index = store.index_by_id(ingest.all_products())
    editors = store.index_by_id(tables["editors"])
    if args.editor not in editors:
        raise SystemExit(f"编辑不存在: {args.editor}")
    record = lifecycle.submit_review(
        product=products_index[args.product],
        editor=editors[args.editor],
        classes=args.classes,
        valid_specs=args.valid_specs,
        status=args.status,
        selected_by=args.selected_by,
        selection_probability=args.selection_probability,
        products_index=products_index,
        moment=args.moment,
    )
    _print(record)


def cmd_withdraw(args) -> None:
    lifecycle.withdraw_review(args.review, moment=args.moment)
    _print(lifecycle.propagate_withdrawal(args.review))


def cmd_publish(args) -> None:
    tables = store.load_raw_tables()
    products_index = store.index_by_id(ingest.all_products())
    decisions = store.read_table(store.decision_path(args.model))
    proposals = lifecycle.propose_adoptions(
        decisions, lifecycle.all_reviews(), products_index, args.model
    )
    result = lifecycle.publish(proposals, args.model, batch_id=args.batch)
    _print({
        "snapshot_id": result["snapshot_id"],
        "published_count": len(result["published"]),
        "blocked_count": len(result["blocked"]),
        "blocked": result["blocked"],
        "published_units_deduped": result["published_units_deduped"],
    })


def cmd_rollback(args) -> None:
    _print(lifecycle.rollback(args.snapshot))


def cmd_diff(args) -> None:
    current = store.read_table(store.decision_path(args.current))
    candidate = store.read_table(store.decision_path(args.candidate))
    products_index = store.index_by_id(ingest.all_products())
    _print(lifecycle.diff_model_versions(current, candidate, products_index))


def cmd_reconcile(args) -> None:
    _print({
        "publication_records": lifecycle.reconcile_publication_records(),
        "model_state": pipeline.cross_check_state(args.model) if args.model else None,
    })


def cmd_models(args) -> None:
    _print({"models": lifecycle_store_models()})


def lifecycle_store_models():
    from . import models
    return models.list_models()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claimdesk", description="包装声明分类与栏目采用工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import", help="导入一批新包装记录")
    p.add_argument("--file", required=True)
    p.add_argument("--batch", required=True)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("train", help="按训练资格规则训练候选模型")
    p.add_argument("--candidate", required=True, choices=("lr-text", "lr-rich"))
    p.add_argument("--tag")
    p.add_argument("--moment", help="时点模拟：只用该时点可知的定稿")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="家族隔离评估并按容限选阈")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="并排比较候选模型")
    p.add_argument("--models", nargs="+", required=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("apply", help="对全目录生成三类决定")
    p.add_argument("--model", required=True)
    p.add_argument("--moment")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("explain", help="查看发布或拒判的具体依据")
    p.add_argument("--model", required=True)
    p.add_argument("--product", required=True)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("queue", help="生成每周复核队列与抽核机会")
    p.add_argument("--model", required=True)
    p.add_argument("--week", required=True)
    p.add_argument("--audit-share", type=float, default=config.DEFAULT_AUDIT_SHARE)
    p.add_argument("--audit-probability", type=float, default=config.DEFAULT_AUDIT_PROBABILITY)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("assignments", help="查看某周编辑分配")
    p.add_argument("--week", required=True)
    p.set_defaults(func=cmd_assignments)

    p = sub.add_parser("submit", help="编辑提交临时意见或定稿")
    p.add_argument("--product", required=True)
    p.add_argument("--editor", required=True)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--valid-specs", nargs="+", required=True)
    p.add_argument("--status", choices=("temporary", "final"), required=True)
    p.add_argument("--selected-by", choices=("random_audit", "suspect_queue"), required=True)
    p.add_argument("--selection-probability", type=float)
    p.add_argument("--moment")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("withdraw", help="撤销定稿并传播证据失效范围")
    p.add_argument("--review", required=True)
    p.add_argument("--moment")
    p.set_defaults(func=cmd_withdraw)

    p = sub.add_parser("publish", help="确认发布（先写可回退快照）")
    p.add_argument("--model", required=True)
    p.add_argument("--batch")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("rollback", help="按快照回退发布")
    p.add_argument("--snapshot", required=True)
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("diff", help="比较现行与候选模型的栏目增删")
    p.add_argument("--current", required=True)
    p.add_argument("--candidate", required=True)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("reconcile", help="模型输出/人审/栏目三方核对")
    p.add_argument("--model")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("models", help="列出已训练模型")
    p.set_defaults(func=cmd_models)
    return parser


def main(argv=None) -> None:
    config.ensure_dirs()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
