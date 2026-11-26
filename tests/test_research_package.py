import copy
from pathlib import Path
import tempfile
import unittest

from scripts import build_research_package as r


def fixture():
    rows = []
    for day, ft, guardian, threat in [("2024-01-01", 1, 1, 2), ("2024-01-08", 4, 4, 0)]:
        for view, count in [("ft", ft), ("guardian", guardian), ("combined", ft + guardian)]:
            rows.append({"week_start": day, "view": view, "timing": "publication",
                         "news_features_available": True, "candidate_count": count,
                         "counts": {"relevant": count, "threat": threat if view == "combined" else 0, "disruption": 0}})
    finance = {k: {} for k in ("primary", "bootstrap", "bootstrap_primary_block_weeks", "panels_common_weeks", "years", "crisis_exclusion")}
    finance.update(final_test_scored=True, retuned_from_final_test=False)
    return {"weekly": rows, "weekly_summary": {"classified_documents": 10, "source_documents": {"ft": 5, "guardian": 5}},
            "nlp": {"methods": {m: {"overall": {}, "without_dominant_component": {}} for m in r.METHODS}}, "finance": finance}


class ResearchPackageTests(unittest.TestCase):
    def test_quarter_share_weights_candidate_counts(self):
        q = r.report_data(fixture())["coverage"][0]
        self.assertEqual(q["candidate_count"], 10)
        self.assertEqual(q["threat_share"], .2)  # Weekly mean would incorrectly be .5.
        self.assertEqual(q["weeks"], 2)

    def test_unavailable_and_other_timing_rows_excluded(self):
        data = fixture()
        for timing, available, day in [("conservative", True, "2024-01-01"), ("publication", False, "2024-01-01"), ("publication", True, "2018-12-31")]:
            extra = copy.deepcopy(data["weekly"][2])
            extra.update(timing=timing, news_features_available=available, week_start=day, candidate_count=999)
            data["weekly"].append(extra)
        self.assertEqual(r.report_data(data)["coverage"][0]["candidate_count"], 10)

    def test_source_count_mismatch_rejected(self):
        data = fixture(); data["weekly"][0]["candidate_count"] += 1
        with self.assertRaisesRegex(ValueError, "do not reconcile"):
            r.report_data(data)

    def test_zero_candidate_quarter_is_unknown_share(self):
        data = fixture()
        for row in data["weekly"]:
            row["candidate_count"] = 0
            row["counts"] = dict.fromkeys(r.TARGETS, 0)
        q = r.report_data(data)["coverage"][0]
        self.assertIsNone(q["threat_share"])
        self.assertIsNone(q["disruption_share"])

    def test_unscored_or_retuned_evaluation_rejected(self):
        for key, value in [("final_test_scored", False), ("retuned_from_final_test", True)]:
            data = fixture(); data["finance"][key] = value
            with self.assertRaisesRegex(ValueError, "completed frozen"):
                r.report_data(data)

    def test_changed_frozen_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); p = root / "result.json"; p.write_text('{"mae": 2.7}')
            expected = {p.name: r.digest(p)}
            r.verify_hashes(root, expected)
            p.write_text('{"mae": 1.0}')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                r.verify_hashes(root, expected)


if __name__ == "__main__":
    unittest.main()
