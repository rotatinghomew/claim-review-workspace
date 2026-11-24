"""特征与模型：多标签学习、每份/每百克折算、缺糖不插补。"""
import numpy as np

from claimdesk import store
from claimdesk.features import PackagingFeatures
from claimdesk.labels import CLASSES, label_matrix, training_labels
from claimdesk.models import CANDIDATES, ClaimClassifier
from tests._base import IsolatedStateCase


class FeatureModelTests(IsolatedStateCase):
    def setUp(self):
        super().setUp()
        tables = store.load_raw_tables()
        self.products = tables["products"]
        self.reviews = tables["reviews"]
        self.index = store.index_by_id(self.products)
        self.labels = training_labels(self.products, self.reviews, self.index)
        self.records = [p for p in self.products if p["id"] in self.labels]
        self.y = np.asarray(label_matrix(self.records, self.labels))

    def test_serving_basis_normalization(self):
        feats = PackagingFeatures().fit(self.records)
        serving = next(p for p in self.products if p["basis"] == "perServing")
        explanation = feats.explain_flags(serving)
        if serving["sugar_g"] is not None:
            expected = round(serving["sugar_g"] * 100.0 / serving["basis_g"], 3)
            self.assertAlmostEqual(
                explanation["sugar_per_100g_derived"], expected, places=3
            )

    def test_missing_sugar_is_flag_not_zero(self):
        feats = PackagingFeatures().fit(self.records)
        missing = next(p for p in self.products if p["sugar_g"] is None)
        explanation = feats.explain_flags(missing)
        self.assertTrue(explanation["sugar_missing"])
        self.assertIsNone(explanation["sugar_per_100g_derived"])

    def test_both_candidates_train_and_predict_shape(self):
        for candidate in CANDIDATES:
            model = ClaimClassifier(candidate).fit(self.records, self.y)
            proba = model.predict_proba(self.records[:5])
            self.assertEqual(proba.shape, (5, len(CLASSES)))
            pred = model.predict(self.records[:5])
            self.assertEqual(pred.shape, (5, len(CLASSES)))
            self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_multi_label_possible(self):
        # 人为制造一条双标签记录，确认 OneVsRest 可以同时输出两个 1
        records = self.records[:40]
        y = self.y[:40].copy()
        y[0, 0] = 1
        y[0, 1] = 1
        model = ClaimClassifier("lr-rich").fit(records, y)
        pred = model.predict(records, np.full(len(CLASSES), 0.01))
        self.assertEqual(pred[0].sum(), len(CLASSES))

    def test_save_load_roundtrip(self):
        model = ClaimClassifier("lr-text").fit(self.records, self.y)
        model_dir = model.save("unit-model", {"note": "roundtrip"})
        reloaded = ClaimClassifier.load if hasattr(ClaimClassifier, "load") else None
        from claimdesk.models import load_model
        again = load_model("unit-model")
        np.testing.assert_allclose(
            again.predict_proba(self.records[:3]),
            model.predict_proba(self.records[:3]),
        )
        self.assertTrue(model_dir.exists())
