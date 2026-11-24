"""重训纪律回归：撤销意见不被带回，历史评估按当时定稿解释。"""
from claimdesk import lifecycle, pipeline, store
from claimdesk.labels import parse_ts, training_labels
from tests._base import IsolatedStateCase


class RetrainDisciplineTests(IsolatedStateCase):
    def setUp(self):
        super().setUp()
        tables = store.load_raw_tables()
        self.products = tables["products"]
        self.products_index = store.index_by_id(self.products)

    def test_withdrawn_review_absent_after_retrain_labels(self):
        # 选一条正式定稿，其 valid_specs 没有其他正式定稿覆盖，撤销后标签必须消失
        reviews = lifecycle.all_reviews()
        target = None
        for review in reviews:
            if review["status"] != "final" or review["withdrawn_at"]:
                continue
            spec = review["product_id"]
            others = [
                r for r in reviews
                if r["id"] != review["id"] and r["status"] == "final"
                and not r["withdrawn_at"] and spec in r["valid_specs"]
            ]
            if not others:
                target = review
                break
        self.assertIsNotNone(target)
        before = training_labels(self.products, reviews, self.products_index)
        cls = target["classes"][0]
        self.assertIn(cls, before[target["product_id"]])

        lifecycle.withdraw_review(target["id"])
        after = training_labels(
            self.products, lifecycle.all_reviews(), self.products_index
        )
        self.assertNotIn(cls, after.get(target["product_id"], set()))

    def test_historical_point_in_time_keeps_then_known_truth(self):
        # 同一条撤销定稿：在撤销之前的时点训练仍然可见（历史表现按当时可知解释）
        review = next(
            r for r in lifecycle.all_reviews()
            if r["status"] == "final" and r["withdrawn_at"]
        )
        moment = parse_ts(review["withdrawn_at"])
        from datetime import timedelta
        earlier = moment - timedelta(days=1)
        labels = training_labels(
            self.products, lifecycle.all_reviews(), self.products_index,
            moment=earlier,
        )
        self.assertIn(review["product_id"], labels)
        # 当前时点则不可见
        labels_now = training_labels(
            self.products, lifecycle.all_reviews(), self.products_index
        )
        surviving = [
            r for r in lifecycle.all_reviews()
            if r["id"] != review["id"] and r["status"] == "final"
            and not r["withdrawn_at"]
            and review["product_id"] in r["valid_specs"]
        ]
        if not surviving:
            self.assertNotIn(review["classes"][0],
                             labels_now.get(review["product_id"], set()))

    def test_retrain_metadata_records_exclusions(self):
        result = pipeline.train_candidate("lr-text", tag="guard")
        meta = pipeline.models.model_metadata(result["model_version"])
        training = meta["training"]
        self.assertGreaterEqual(training["excluded_temporary_reviews"], 25)
        self.assertTrue(training["unlabeled_are_unlabeled_not_negative"])
        self.assertEqual(
            set(training["labeled_positives_per_class"]),
            {"未添加某种糖", "总糖含量声称", "无关口味宣传", "无法判断"},
        )
