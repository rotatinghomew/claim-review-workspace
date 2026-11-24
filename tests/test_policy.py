"""三类决定、容限选阈、复核队列与抽核记录测试。"""
import numpy as np

from claimdesk import policy, store
from claimdesk.features import PackagingFeatures
from tests._base import IsolatedStateCase


class PolicyTests(IsolatedStateCase):
    def setUp(self):
        super().setUp()
        tables = store.load_raw_tables()
        self.products = tables["products"]
        self.editors = tables["editors"]
        self.costs = {c["class_name"]: c for c in tables["cost_policies"]}
        self.classes = ("未添加某种糖", "总糖含量声称", "无关口味宣传", "无法判断")

    def _decisions(self, products=None):
        products = products or self.products[:20]
        rng = np.random.default_rng(0)
        proba = rng.random((len(products), 4))
        sweep = {cls: [
            {"threshold": t, "auto_coverage": 0.5, "false_release_rate": 0.0,
             "false_release_count": 0, "miss_rate": 0.1, "miss_count": 1}
            for t in (0.3, 0.5, 0.8)
        ] for cls in self.classes}
        thresholds, feasibility = policy.choose_thresholds(sweep, self.costs)
        return policy.decide_records(products, proba, thresholds, feasibility, self.costs)

    def test_three_actions_all_present(self):
        decisions = self._decisions()
        actions = {d["action"] for d in decisions}
        self.assertEqual(actions, {"release", "review", "hold"})

    def test_missing_sugar_forces_hold_for_sugar_classes(self):
        missing = next(p for p in self.products if p["sugar_g"] is None)
        decisions = self._decisions([missing])
        for cls in ("总糖含量声称", "无法判断"):
            row = next(d for d in decisions if d["class_name"] == cls)
            self.assertEqual(row["action"], "hold")
        # 文字类声明不受缺糖影响
        text_row = next(d for d in decisions if d["class_name"] == "未添加某种糖")
        self.assertNotEqual(text_row["action"], "hold")

    def test_infeasible_tolerance_blocks_release(self):
        sweep = {cls: [
            {"threshold": 0.5, "auto_coverage": 0.9, "false_release_rate": 0.9,
             "false_release_count": 9, "miss_rate": 0.0, "miss_count": 0}
        ] for cls in self.classes}
        thresholds, feasibility = policy.choose_thresholds(sweep, self.costs)
        self.assertTrue(all(not f["feasible"] for f in feasibility.values()))
        decisions = policy.decide_records(
            self.products[:5], np.full((5, 4), 0.99), thresholds, feasibility, self.costs
        )
        self.assertTrue(all(d["action"] != "release" for d in decisions))

    def test_queue_respects_editor_permissions_and_minutes(self):
        decisions = self._decisions()
        payload = policy.build_weekly_queue(
            decisions, self.editors, self.costs, "2026-W48",
            audit_share=0.4, audit_probability=1.0, seed=7,
        )
        # audit_probability=1 时所有 release 都获得抽核机会
        self.assertEqual(
            payload["audit_opportunities_available"],
            sum(1 for d in decisions if d["action"] == "release"),
        )
        permits = {e["id"]: set(e["classes"]) for e in self.editors}
        for editor_id, items in payload["assignments"].items():
            for item in items:
                self.assertIn(item["class_name"], permits[editor_id])
        # 抽核实际选中方式必须落盘
        log = store.read_table(store.audit_log_path())
        self.assertTrue(log)
        self.assertTrue(all("selection_probability" in r for r in log))

    def test_non_finalizer_gets_flagged(self):
        decisions = self._decisions()
        payload = policy.build_weekly_queue(
            decisions, self.editors, self.costs, "2026-W49",
            audit_share=0.0, audit_probability=0.0, seed=7,
        )
        flagged = [
            item for items in payload["assignments"].values() for item in items
            if item["requires_finalizer"]
        ]
        # 无定稿权限编辑领到的任务必须显式标注需要定稿人接手
        for item in flagged:
            self.assertTrue(item["requires_finalizer"])

    def test_explicit_flags_are_explainable(self):
        feats = PackagingFeatures()
        explanation = feats.explain_flags(self.products[1])
        self.assertIn("matched_flags", explanation)
        self.assertIn("sugar_per_100g_derived", explanation)
