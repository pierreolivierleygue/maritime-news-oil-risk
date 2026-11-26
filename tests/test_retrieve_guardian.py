"""Data-integrity checks using synthetic API responses; no network or credentials."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import retrieve_guardian as guardian


def response(total, page):
    start = (page - 1) * guardian.PAGE_SIZE
    return {
        "status": "ok", "total": total, "pages": (total + guardian.PAGE_SIZE - 1) // guardian.PAGE_SIZE,
        "currentPage": page, "pageSize": guardian.PAGE_SIZE, "orderBy": "oldest",
        "results": [{"id": f"test/{n}", "webTitle": f"Story {n}", "type": "article",
                     "webPublicationDate": "2019-01-02T10:00:00Z", "fields": {"bodyText": "Text"}}
                    for n in range(start, min(total, start + guardian.PAGE_SIZE))],
    }


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)

    def test_pagination_and_offline_resume(self):
        calls = []
        def client(params):
            calls.append(params)
            return response(101, params["page"])
        payload, records = guardian.collect_month("2019-01", self.cache, client)
        self.assertEqual(len(records), 101)
        self.assertEqual([p["page"] for p in calls], [1, 2])
        self.assertNotIn("api-key", json.dumps(payload))
        cached, again = guardian.collect_month("2019-01", self.cache, lambda p: self.fail("Network called"), True)
        self.assertEqual(payload, cached)
        self.assertEqual(records, again)

    def test_zero_is_complete_but_missing_is_not(self):
        payload, records = guardian.collect_month("2019-01", self.cache, lambda p: response(0, 1))
        self.assertEqual(records, [])
        self.assertEqual(len(payload["pages"]), 1)
        with self.assertRaisesRegex(RuntimeError, "missing complete cache"):
            guardian.collect_month("2019-02", self.cache, None, True)

    def test_truncated_page_never_cached(self):
        partial = response(2, 1)
        partial["results"].pop()
        with self.assertRaisesRegex(RuntimeError, "truncated page"):
            guardian.collect_month("2019-01", self.cache, lambda p: partial)
        self.assertFalse(list(self.cache.iterdir()))

    def test_changing_totals_never_cached(self):
        def client(params):
            return response(101 if params["page"] == 1 else 102, params["page"])
        with self.assertRaisesRegex(RuntimeError, "totals changed"):
            guardian.collect_month("2019-01", self.cache, client)
        self.assertFalse(list(self.cache.iterdir()))

    def test_duplicates_and_wrong_month_rejected(self):
        for corruption in ("duplicate", "date"):
            bad = response(2, 1)
            if corruption == "duplicate":
                bad["results"][1]["id"] = bad["results"][0]["id"]
            else:
                bad["results"][1]["webPublicationDate"] = "2019-02-02T10:00:00Z"
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                guardian.collect_month("2019-01", self.cache, lambda p: bad)
        self.assertFalse(list(self.cache.iterdir()))

    def test_incomplete_and_incompatible_cache_rejected(self):
        payload, _ = guardian.collect_month("2019-01", self.cache, lambda p: response(101, p["page"]))
        partial = copy.deepcopy(payload)
        partial["pages"].pop()
        with self.assertRaisesRegex(RuntimeError, "incomplete pagination"):
            guardian.validate_month(partial, "2019-01")
        payload["params"]["q"] = "another query"
        (self.cache / "2019-01.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, "settings differ"):
            guardian.collect_month("2019-01", self.cache, None, True)


if __name__ == "__main__":
    unittest.main()
