"""Meaningful invariants for exploratory screening and weekly coverage."""

from datetime import date
import unittest

from scripts.analyze_news_corpus import (
    FT, GUARDIAN, combine_versions, jaccard, monday, screen, shingle_set, weekly_rows,
)


class AnalysisTests(unittest.TestCase):
    def test_screen_requires_local_geography_transport_and_risk(self):
        self.assertTrue(screen("", "Tankers face attack threats in the Strait of Hormuz.")["candidate_relevant"])
        self.assertFalse(screen("", "The Red Sea Development Company opened a new hotel.")["candidate_relevant"])
        self.assertFalse(screen("", "Hormuz. " + "ordinary business " * 100 + " Tankers face attack threats.")["candidate_relevant"])

    def test_negated_and_historical_language_are_not_event_labels(self):
        result = screen("", "No disruption affected shipping in the Red Sea; attacks occurred years ago.")
        self.assertTrue(result["candidate_relevant"])
        self.assertIsNone(result["relevant"])
        self.assertIsNone(result["threat"])
        self.assertIsNone(result["disruption"])

    def test_metadata_versions_share_document_but_conflicts_are_not_silently_chosen(self):
        base = {"source": FT, "document_id": "123", "body_hash": "same",
                "publication_date": "2020-01-02", "title": "Title", "record_version_hash": "b"}
        rows = combine_versions([base, {**base, "record_version_hash": "a"}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["version_hashes"], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "conflict"):
            combine_versions([base, {**base, "body_hash": "different"}])

    def test_shingles_ignore_case_but_do_not_merge_short_texts(self):
        text = " ".join(f"word{n}" for n in range(80))
        self.assertEqual(jaccard(shingle_set(text), shingle_set(text.upper())), 1)
        self.assertEqual(shingle_set("A short briefing"), set())
        self.assertEqual(jaccard(shingle_set(text), shingle_set(" ".join(f"other{n}" for n in range(80)))), 0)

    def test_weekly_counts_deduplicate_within_week_and_keep_later_republication(self):
        base = {"source": FT, "exact_group_id": "same_text", "near_family_id": "same_family",
                "candidate_relevant": True}
        docs = [{**base, "publication_date": day} for day in ("2020-01-06", "2020-01-07", "2020-01-13")]
        rows = weekly_rows(docs, [FT], {2020: "partial"}, set())
        first = next(r for r in rows if r["week_start"] == "2020-01-06")
        second = next(r for r in rows if r["week_start"] == "2020-01-13")
        self.assertEqual(first["observed_document_count"], 2)
        self.assertEqual(first["exact_text_count"], 1)
        self.assertEqual(first["candidate_share"], 1)
        self.assertEqual(second["exact_text_count"], 1)
        self.assertEqual(first["collection_status"], "known_annual_gap_accepted")

    def test_empty_incomplete_week_is_not_confirmed_zero(self):
        rows = weekly_rows([], [FT, GUARDIAN], {2020: "partial"}, {"2020-01"})
        ft = next(r for r in rows if r["source"] == FT and r["week_start"] == "2020-01-06")
        guardian = next(r for r in rows if r["source"] == GUARDIAN and r["week_start"] == "2020-01-06")
        self.assertIsNone(ft["candidate_share"])
        self.assertEqual(ft["zero_interpretation"], "no_observed_records_completeness_not_verified")
        self.assertEqual(guardian["zero_interpretation"], "confirmed_no_query_matches")
        february = next(r for r in rows if r["source"] == GUARDIAN and r["week_start"] == "2020-01-27")
        self.assertEqual(february["collection_status"], "collection_unverified")

    def test_boundary_weeks_are_explicit_and_sources_remain_separate(self):
        self.assertEqual(monday(date(2019, 1, 1)), date(2018, 12, 31))
        rows = weekly_rows([], [FT, GUARDIAN], {}, set())
        self.assertEqual(len(rows), 366 * 2)
        self.assertTrue(rows[0]["partial_boundary_week"])
        self.assertTrue(rows[-1]["partial_boundary_week"])
        self.assertEqual(rows[-1]["observed_end"], "2025-12-31")


if __name__ == "__main__":
    unittest.main()
