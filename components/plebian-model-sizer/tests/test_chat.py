from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from plebian_model_sizer.chat import REQUEST_SCHEMA, RESPONSE_SCHEMA, recommend_chat
from plebian_model_sizer.resources import GIB
from test_sizing import snapshot


def request():
    return {"schema": REQUEST_SCHEMA, "models": [
        {"id": "qwen3.8:27b", "model_bytes": 17 * GIB,
         "parameters": 27_300_000_000, "installed": True, "runtime_supported": True},
        {"id": "qwen3:4b-instruct-2507-q4_K_M", "model_bytes": 2_497_293_803,
         "parameters": 4_000_000_000, "installed": True, "runtime_supported": True}]}


class AvatarChatSizingTests(unittest.TestCase):
    def test_keeps_desktop_headroom_and_prefers_fitting_installed_model(self):
        resources = snapshot()
        resources["ram_available_bytes"] = 32 * GIB
        report = recommend_chat(request(), resources)
        self.assertEqual(report["schema"], RESPONSE_SCHEMA)
        self.assertEqual(report["resource_source"], "live")
        self.assertEqual(report["shortlist"], ["qwen3:4b-instruct-2507-q4_K_M"])
        self.assertEqual(report["provisional_candidate"], report["shortlist"][0])
        self.assertIsNone(report["selected_model"])
        self.assertFalse(report["qualification_eligible"])
        self.assertEqual(report["candidates"][0]["verdict"], "does-not-fit")

    def test_unknown_or_stale_resources_never_fit(self):
        resources = snapshot()
        resources["ram_available_bytes"] = None
        self.assertEqual(recommend_chat(request(), resources)["shortlist"], [])
        resources = snapshot()
        resources["observed_at"] = (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()
        self.assertEqual(recommend_chat(request(), resources)["shortlist"], [])

    def test_caller_assertions_and_invalid_catalogs_do_not_select(self):
        catalog = request()
        catalog["models"][1]["installed"] = False
        self.assertNotIn(catalog["models"][1]["id"], recommend_chat(catalog, snapshot())["shortlist"])
        for mutation in (
                lambda r: r["models"][1].update(model_bytes=True),
                lambda r: r["models"][1].update(installed=None),
                lambda r: r["models"][1].update(id=r["models"][0]["id"]),
                lambda r: r.update(schema="other")):
            invalid = deepcopy(request())
            mutation(invalid)
            with self.assertRaises(ValueError):
                recommend_chat(invalid, snapshot())
