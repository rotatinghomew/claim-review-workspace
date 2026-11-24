"""评估纪律：家族隔离、按类/按渠道口径、抽核加权估计。"""
import numpy as np

from claimdesk import evaluate, store
from claimdesk.labels import label_matrix, training_labels
from claimdesk.models import ClaimClassifier
from tests._base import IsolatedStateCase


class EvaluationTests(IsolatedStateCase):
    def setUp(self):
        super().setUp()
        tables = store.load_raw_tables()
        self.products = tables["products"]
        self.reviews = tables["reviews"]
        self.index = store.index_by_id(self.products)
        self.labels = training_labels(self.products, self.reviews, self.index)

    def test_no_family_leak_in_holdout(self):
        split = evaluate.family_holdout_split(self.products)
        train_families = {self.index[i]["family"] for i in split.train_ids}
        eval_families = {self.index[i]["family"] for i in split.eval_ids}
        self.assertFalse(train_families & eval_families)

    def test_temporal_split_keeps_families_apart_by_record(self):
        split = evaluate.temporal_split(self.products)
        self.assertFalse(set(split.train_ids) & set(split.eval_ids))
        # 评估侧全部为更高版本
        self.assertTrue(all(self.index[i]["version"] == 3 for i in split.eval_ids))

    def test_per_class_metrics_structure(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1]])
        y_pred = np.array([[1, 0], [0, 0], [0, 1]])
        report = evaluate.per_class_reliability(y_true, y_pred, classes=("a", "b"))
        self.assertEqual(report["a"]["false_release_count"], 0)
        self.assertEqual(report["a"]["miss_count"], 1)
        self.assertEqual(report["b"]["false_release_count"], 0)

    def test_unlabeled_channel_marked_not_guessed(self):
        # ch3 在合成资料中没有任何合格定稿
        tables = store.load_raw_tables()
        ch3 = [p for p in self.products if p["channel"] == "ch3"]
        y_true = np.zeros((len(ch3), 4), dtype=int)
        y_pred = np.ones((len(ch3), 4), dtype=int)
        report = evaluate.per_channel_reliability(ch3, y_true, y_pred)
        self.assertEqual(report["ch3"]["status"], "no_eligible_labels")

    def test_full_evaluation_report_has_required_views(self):
        ids = evaluate.labeled_only([p["id"] for p in self.products], self.labels)
        records = [self.index[i] for i in ids]
        y = np.asarray(label_matrix(records, self.labels))
        model = ClaimClassifier("lr-text").fit(records, y)
        proba = model.predict_proba(records)
        sweep = evaluate.threshold_sweep(proba, y)
        self.assertEqual(set(sweep), set(("未添加某种糖", "总糖含量声称",
                                          "无关口味宣传", "无法判断")))
        for rows in sweep.values():
            self.assertTrue(all("false_release_rate" in r for r in rows))

    def test_ipw_uses_only_random_audits(self):
        release_rows = [
            {"product_id": "p1", "predicted_classes": ["总糖含量声称"]},
        ]
        # 定向疑点意见不得进入加权
        suspect = [{
            "id": "lq", "product_id": "p1", "status": "final",
            "classes": ["无关口味宣传"], "selected_by": "suspect_queue",
            "selection_probability": None, "valid_specs": ["p1"],
            "known_at": "2026-03-01T00:00:00+08:00", "withdrawn_at": None,
        }]
        report = evaluate.ipw_release_error(release_rows, suspect, self.index)
        self.assertIsNone(report["总糖含量声称"]["ipw_false_release_rate"])
        self.assertEqual(report["_meta"]["suspect_queue_only_audits_excluded"], 1)

    def test_ipw_flags_error_with_weight(self):
        random_review = [{
            "id": "la", "product_id": "p1", "status": "final",
            "classes": ["无关口味宣传"], "selected_by": "random_audit",
            "selection_probability": 0.15, "valid_specs": ["p1"],
            "known_at": "2026-03-01T00:00:00+08:00", "withdrawn_at": None,
        }]
        release_rows = [{"product_id": "p1",
                         "predicted_classes": ["总糖含量声称"]}]
        report = evaluate.ipw_release_error(release_rows, random_review, self.index)
        # 抽核真值不含该类 => 加权错误率为 1.0
        self.assertEqual(report["总糖含量声称"]["ipw_false_release_rate"], 1.0)
