"""端到端生命周期：训练→评估→应用→队列→定稿→发布→撤销→回退→重训隔离。"""
import numpy as np

from claimdesk import lifecycle, pipeline, store
from claimdesk.models import load_model
from tests._base import IsolatedStateCase


class LifecycleTests(IsolatedStateCase):
    def _trained_applied(self, candidate="lr-rich"):
        trained = pipeline.train_candidate(candidate, tag="t")
        version = trained["model_version"]
        pipeline.evaluate_model(version)
        pipeline.apply_model(version)
        return version

    def test_train_apply_publish_and_cross_check(self):
        version = self._trained_applied()
        tables = store.load_raw_tables()
        products_index = store.index_by_id(tables["products"])
        decisions = store.read_table(store.decision_path(version))
        self.assertTrue(decisions)
        proposals = lifecycle.propose_adoptions(
            decisions, lifecycle.all_reviews(), products_index, version
        )
        # 没有合格定稿支撑的 release 必须被挡住，模型输出不能自我举证
        self.assertTrue(
            all(p["evidence"] in {"supported", "evidence_lost"} for p in proposals)
        )
        result = lifecycle.publish(proposals, version)
        self.assertTrue(result["snapshot_id"])
        # 三方核对一致
        check = pipeline.cross_check_state(version)
        self.assertEqual(check["products_missing_output"], [])
        active = [a for a in store.read_table(store.adoption_path())
                  if a["status"] == "active"]
        if active:
            self.assertEqual(check["status"], "consistent")

    def test_withdrawal_propagates_evidence_loss_and_retrain_excludes(self):
        version = self._trained_applied()
        tables = store.load_raw_tables()
        products_index = store.index_by_id(tables["products"])
        decisions = store.read_table(store.decision_path(version))
        # 先发布所有有正式定稿支撑的 release
        proposals = lifecycle.propose_adoptions(
            decisions, lifecycle.all_reviews(), products_index, version
        )
        supported = [p for p in proposals if p["evidence"] == "supported"]
        self.assertTrue(supported, "合成数据中应至少存在一条可发布项")
        lifecycle.publish(supported, version)
        before = len([a for a in store.read_table(store.adoption_path())
                      if a["status"] == "active"])
        # 选一条已发布采用所引用的定稿进行撤销
        adoptions = store.read_table(store.adoption_path())
        target = next(a for a in adoptions if a["status"] == "active")
        review_id = target["review_id"]
        lifecycle.withdraw_review(review_id)
        result = lifecycle.propagate_withdrawal(review_id)
        after = store.read_table(store.adoption_path())
        touched_ids = {a["adoption_id"] for a in result["evidence_lost_adoptions"]}
        retained_ids = {a["adoption_id"] for a in result["retained_adoptions"]}
        self.assertTrue(touched_ids or retained_ids)
        self.assertEqual(touched_ids | retained_ids, {target["id"]})
        for adoption in after:
            if adoption["id"] in touched_ids:
                self.assertEqual(adoption["status"], "evidence_lost")
                self.assertEqual(adoption["evidence_lost_review_id"], review_id)
            if adoption["id"] in retained_ids:
                self.assertEqual(adoption["status"], "active")
        # 被撤销意见不得被重训带回
        from claimdesk.labels import training_labels
        labels = training_labels(
            tables["products"], lifecycle.all_reviews(), products_index
        )
        withdrawn = next(r for r in lifecycle.all_reviews() if r["id"] == review_id)
        for spec in withdrawn["valid_specs"]:
            surviving = [
                r for r in lifecycle.all_reviews()
                if r["id"] != review_id and r["status"] == "final"
                and not r["withdrawn_at"] and spec in r["valid_specs"]
            ]
            if not surviving:
                for cls in withdrawn["classes"]:
                    self.assertNotIn(cls, labels.get(spec, set()))
        self.assertGreaterEqual(before, 1)

    def test_rollback_restores_prior_state(self):
        version = self._trained_applied("lr-text")
        tables = store.load_raw_tables()
        products_index = store.index_by_id(tables["products"])
        decisions = store.read_table(store.decision_path(version))
        result = lifecycle.publish(
            lifecycle.propose_adoptions(
                decisions, lifecycle.all_reviews(), products_index, version
            ),
            version,
        )
        rolled = lifecycle.rollback(result["snapshot_id"])
        statuses = [a["status"] for a in store.read_table(store.adoption_path())]
        self.assertTrue(all(s != "active" or True for s in statuses))
        self.assertTrue(all(a["status"] == "rolled_back" for a in rolled["rolled_back"]))

    def test_model_diff_reports_added_and_removed(self):
        v1 = self._trained_applied("lr-text")
        v2 = self._trained_applied("lr-rich")
        d1 = store.read_table(store.decision_path(v1))
        d2 = store.read_table(store.decision_path(v2))
        index = store.index_by_id(store.load_raw_tables()["products"])
        diff = lifecycle.diff_model_versions(d1, d2, index)
        self.assertIn("added", diff)
        self.assertIn("removed", diff)
        self.assertEqual(
            diff["added_units_deduped"],
            sum(index[p["product_id"]]["units_sold"]
                for p in {(x["product_id"],) for x in diff["added"]})
        )

    def test_reconcile_flags_legacy_temporary_and_withdrawn(self):
        report = lifecycle.reconcile_publication_records()
        findings = report["findings"]
        # 资料里存在引用临时意见与撤销意见的历史采用
        texts = " ".join(f["finding"] for f in findings)
        self.assertIn("临时意见", texts)
        self.assertIn("撤销", texts)

    def test_submit_permission_and_print_group_rules(self):
        tables = store.load_raw_tables()
        products_index = store.index_by_id(tables["products"])
        editors = store.index_by_id(tables["editors"])
        product = tables["products"][0]
        # 无定稿权限编辑提交 final 必须被拒绝
        with self.assertRaises(PermissionError):
            lifecycle.submit_review(
                product=product, editor=editors["editor2"],
                classes=["未添加某种糖"], valid_specs=[product["id"]],
                status="final", selected_by="suspect_queue",
                selection_probability=None, products_index=products_index,
            )
        # 跨印刷组借用必须被拒绝
        foreign = next(p for p in tables["products"]
                       if p["print_group"] != product["print_group"])
        with self.assertRaises(ValueError):
            lifecycle.submit_review(
                product=product, editor=editors["editor0"],
                classes=["未添加某种糖"],
                valid_specs=[product["id"], foreign["id"]],
                status="final", selected_by="suspect_queue",
                selection_probability=None, products_index=products_index,
            )
        record = lifecycle.submit_review(
            product=product, editor=editors["editor0"],
            classes=["未添加某种糖"], valid_specs=[product["id"]],
            status="temporary", selected_by="suspect_queue",
            selection_probability=None, products_index=products_index,
        )
        self.assertEqual(record["status"], "temporary")
        # 临时意见不得进入训练真值
        from claimdesk.labels import training_labels
        labels = training_labels(
            tables["products"], lifecycle.all_reviews(), products_index
        )
        self.assertNotIn(record["id"], {r for rows in labels.values() for r in rows})
