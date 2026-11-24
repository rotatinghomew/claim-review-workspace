"""导入校验与 HTTP 路由测试（直接调用 ASGI 路由函数，不依赖额外网络库）。"""
import json

from claimdesk import api, ingest, store
from tests._base import IsolatedStateCase


class ImportTests(IsolatedStateCase):
    def test_import_rejects_bad_and_accepts_good(self):
        good = {
            "id": "px-new", "family": "fx", "version": 1,
            "valid_from": "2026-06-01", "channel": "ch0",
            "text": "未添加蔗糖", "ingredients": "水",
            "basis": "per100g", "basis_g": 100, "sugar_g": None,
            "units_sold": 5, "known_at": "2026-06-01T00:00:00+08:00",
            "print_group": "print-new",
        }
        bad_missing = {"id": "px-bad", "family": "fy"}
        bad_dup = dict(good)
        from claimdesk import config
        source = config.STATE_DIR / "incoming.jsonl"
        source.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in (good, bad_missing, bad_dup)),
            encoding="utf-8",
        )
        report = ingest.import_products(source, "batch-1")
        self.assertEqual(report["received"], 3)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(len(report["rejected"]), 2)
        # 原始表未被修改
        self.assertNotIn("px-new", {p["id"] for p in store.load_raw_tables()["products"]})
        staged = ingest.staged_products("batch-1")
        self.assertEqual([p["id"] for p in staged], ["px-new"])

    def test_api_routes_share_service_layer(self):
        health = api.health()
        self.assertEqual(health["status"], "ok")
        models = api.list_all_models()
        self.assertIn("models", models)
        # 未知模型的决定查询必须 404，而不是静默成功
        import fastapi
        with self.assertRaises(fastapi.HTTPException):
            api.explain("nope", "p0")
