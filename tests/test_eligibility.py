"""训练资格与意见适用性规则测试。"""
from datetime import datetime

from claimdesk import store
from claimdesk.labels import (
    applicability,
    independent_support,
    is_active_at,
    parse_ts,
    training_labels,
)
from tests._base import IsolatedStateCase


class EligibilityTests(IsolatedStateCase):
    def setUp(self):
        super().setUp()
        tables = store.load_raw_tables()
        self.products = tables["products"]
        self.reviews = tables["reviews"]
        self.products_index = store.index_by_id(self.products)

    def test_temporary_never_truth(self):
        labels = training_labels(self.products, self.reviews, self.products_index)
        temp_products = {
            r["product_id"] for r in self.reviews if r["status"] == "temporary"
        }
        # 临时意见覆盖的规格若没有其他正式定稿，就不得出现在训练标签中
        for pid in temp_products:
            other_final = [
                r for r in self.reviews
                if r["product_id"] == pid and r["status"] == "final"
                and not r["withdrawn_at"]
            ]
            if not other_final:
                self.assertNotIn(pid, labels)

    def test_withdrawn_inactive_now_but_active_before(self):
        withdrawn = next(
            r for r in self.reviews if r["withdrawn_at"] and r["status"] == "final"
        )
        self.assertFalse(is_active_at(withdrawn))
        # 撤销时点之前、定稿之后：该意见在当时属于可知且有效证据
        from datetime import timedelta
        moment = parse_ts(withdrawn["withdrawn_at"]) - timedelta(days=1)
        self.assertTrue(moment > parse_ts(withdrawn["known_at"]))
        self.assertTrue(is_active_at(withdrawn, moment))

    def test_temporary_review_never_active(self):
        temporary = next(r for r in self.reviews if r["status"] == "temporary")
        self.assertFalse(is_active_at(temporary))
        self.assertFalse(is_active_at(temporary, parse_ts("2025-01-01T00:00:00+08:00")))

    def test_point_in_time_unknown_reviews_excluded(self):
        early = datetime.fromisoformat("2026-01-02T00:00:00+08:00")
        labels = training_labels(self.products, self.reviews,
                                 self.products_index, moment=early)
        later_reviewed = {
            r["product_id"] for r in self.reviews
            if parse_ts(r["known_at"]) > early
        }
        # 当时尚不可知的定稿不能进入该时点的训练真值
        for pid in later_reviewed:
            self.assertNotIn(pid, labels)

    def test_cross_print_group_rejected(self):
        review = next(r for r in self.reviews if len(r["valid_specs"]) > 1)
        foreign = next(
            p for p in self.products
            if p["print_group"] != self.products_index[review["product_id"]]["print_group"]
        )
        check = applicability(review, foreign, self.products_index)
        self.assertFalse(check["applicable_for_training"])
        self.assertFalse(check["applicable_for_adoption"])

    def test_superseded_spec_blocks_adoption_but_keeps_training(self):
        # 取一个有正式定稿覆盖的 version=1 规格：家族内存在更高版本 => 已分版
        old = next(
            p for p in self.products
            if p["version"] == 1 and any(
                r["status"] == "final" and p["id"] in r["valid_specs"]
                for r in self.reviews
            )
        )
        review = next(
            r for r in self.reviews
            if old["id"] in r["valid_specs"] and r["status"] == "final"
        )
        check = applicability(review, old, self.products_index)
        self.assertTrue(check["superseded"])
        self.assertTrue(check["applicable_for_training"])
        self.assertFalse(check["applicable_for_adoption"])

    def test_independent_support_after_withdrawal(self):
        # 构造两条支撑同一规格/类别的定稿，撤销其一后仍有独立支撑。
        # 选家族内最新版本，避免分版规则干扰。
        product = next(p for p in self.products if p["version"] == 3)
        r1 = {
            "id": "lx1", "product_id": product["id"], "reviewer": "editor0",
            "classes": ["未添加某种糖"], "status": "final",
            "selected_by": "random_audit", "selection_probability": 0.15,
            "valid_specs": [product["id"]],
            "known_at": "2026-01-21T00:00:00+08:00", "withdrawn_at": None,
        }
        r2 = {**r1, "id": "lx2"}
        support = independent_support(
            product["id"], "未添加某种糖", [r1, r2], self.products_index,
            exclude_review="lx1", moment=None,
        )
        self.assertEqual(support, ["lx2"])

    def test_labels_are_multi_label_sets(self):
        labels = training_labels(self.products, self.reviews, self.products_index)
        for value in labels.values():
            self.assertIsInstance(value, set)

    def test_unlabeled_is_absent_not_negative(self):
        labels = training_labels(self.products, self.reviews, self.products_index)
        unlabeled = [p["id"] for p in self.products if p["id"] not in labels]
        self.assertTrue(unlabeled)  # 90 条无意见记录必须缺席而不是被当作全负
