"""Synthetic checks for chronological, family and transitive report separation."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_news_review import digest, prepare, protected_families, write_rows
from scripts.reserve_news_evaluation import assign, report_links, reserve, validate, save_json


def document(identity, day, family=None, source="The Guardian"):
    return dict(document_id=identity, publication_date=day, near_family_id=family or identity,
                source=source, record_version_hash=identity, version_hashes=[identity],
                body_hash=identity, exact_group_id=identity, title="Example " + identity,
                body_clean="Unrelated synthetic body " + identity, candidate_relevant=False)


def families(docs):
    result = {}
    for d in docs:
        f = result.setdefault(d["near_family_id"], dict(near_family_id=d["near_family_id"],
                               publication_dates=[], exact_group_ids=[]))
        f["publication_dates"].append(d["publication_date"])
        f["exact_group_ids"].append(d["exact_group_id"])
    return list(result.values())


CONFIG = dict(development_start="2019-01-01", evaluation_start="2023-01-01",
              evaluation_end_exclusive="2024-01-01", evaluation_sources=["The Guardian"],
              related_report_rules=[], similarity_days=14, minimum_shingles=20,
              containment_threshold=0.65, prior_episode_review="episodes.json",
              prior_summary_review="summaries.json")


class EvaluationTests(unittest.TestCase):
    def test_transitive_boundary_component_is_excluded(self):
        docs = [document("old", "2022-12-31"), document("middle", "2023-01-01"),
                document("new", "2023-02-01"), document("independent", "2023-03-01")]
        links = [dict(left="old", right="middle"), dict(left="middle", right="new")]
        rows, groups, queue = assign(docs, families(docs), links, CONFIG)
        self.assertEqual([q["document_id"] for q in queue], ["independent"])
        connected = [r for r in rows if r["document_id"] != "independent"]
        self.assertEqual(len({r["split_group_id"] for r in connected}), 1)
        self.assertEqual([r["document_id"] for r in rows if r["nlp_training_eligible"]], ["old"])

    def test_later_followups_stay_held_out_but_are_not_queued(self):
        docs = [document("a", "2023-12-31"), document("b", "2024-01-01")]
        rows, _, queue = assign(docs, families(docs), [dict(left="a", right="b")], CONFIG)
        self.assertEqual(len(queue), 1)
        self.assertEqual({r["partition"] for r in rows}, {"evaluation_reserved"})
        self.assertFalse(any(r["nlp_training_eligible"] for r in rows))

    def test_duplicate_family_is_not_split_and_selection_is_order_independent(self):
        docs = [document("b", "2023-05-01", "shared"), document("a", "2023-05-01", "shared")]
        first = assign(docs, families(docs), [], CONFIG)
        second = assign(list(reversed(docs)), families(docs), [], CONFIG)
        self.assertEqual(first, second)
        self.assertEqual([q["document_id"] for q in first[2]], ["a"])
        # A prior date on any family version protects the entire family.
        fs = families(docs)
        fs[0]["publication_dates"].append("2022-12-31")
        self.assertEqual(assign(docs, fs, [], CONFIG)[2], [])

    def test_prior_packet_exposure_excludes_connected_group(self):
        docs = [document("a", "2023-01-01"), document("b", "2023-02-01")]
        self.assertEqual(assign(docs, families(docs), [dict(left="a", right="b")], CONFIG, ["a"])[2], [])

    def test_cross_source_identical_body_and_rule_links(self):
        docs = [document("a", "2023-01-01"), document("b", "2023-01-01", source="FT"),
                document("c", "2023-02-01")]
        docs[1]["body_hash"] = docs[0]["body_hash"]
        docs[1]["title"] = docs[2]["title"] = "Named vessel"
        config = {**CONFIG, "related_report_rules": [dict(id="vessel", start="2023-01-01",
                  end="2024-01-01", pattern="Named vessel", lead_words=20)]}
        links = report_links(docs, config)
        self.assertEqual({e["reason"] for e in links},
                         {"identical_body_across_sources", "related_report_rule:vessel"})
        self.assertEqual(len(assign(docs, families(docs), links, config)[1]), 1)

    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        batch = root / "data/evaluation/fixture"
        batch.mkdir(parents=True)
        processed = root / "data/processed"
        processed.mkdir(parents=True)
        docs = [document("old", "2022-01-01"), document("new", "2023-01-01")]
        write_rows(processed / "news_documents.jsonl", docs)
        write_rows(processed / "news_review_families.jsonl", families(docs))
        write_rows(processed / "news_screening_review_queue.jsonl", [docs[0]])
        (root / "ANNOTATION_GUIDE.md").write_text("Guide")
        (batch / "guide_v0_2.md").write_text("Guide")
        save_json(batch / "guide_freeze.json", dict(guide_snapshot="guide_v0_2.md",
                  guide_sha256=digest(batch / "guide_v0_2.md"), guide_frozen_at="2026-01-01T00:00:00+00:00"))
        save_json(batch / "selection_config.json", CONFIG)
        save_json(root / "episodes.json", {"groups": []})
        save_json(root / "summaries.json", {"links": []})
        return root, batch

    def test_reservation_reproduces_refuses_overwrite_and_protects_unlabelled_rows(self):
        root, batch = self.fixture()
        reserve(root, "fixture")
        self.assertEqual(validate(root, "fixture")["evaluation_articles"], 1)
        self.assertEqual(protected_families(root)[0], {"new"})
        with self.assertRaisesRegex(ValueError, "Existing reservation"):
            reserve(root, "fixture")
        self.assertFalse((batch / "annotations.jsonl").exists())
        # Normal early development remains available.
        prepare("development", "queue", root)
        with self.assertRaisesRegex(ValueError, "already reserved"):
            prepare("another", "queue", root)

    def test_guide_and_corpus_drift_fail_closed(self):
        root, batch = self.fixture()
        reserve(root, "fixture")
        (batch / "guide_v0_2.md").write_text("Changed guide")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            validate(root, "fixture")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            prepare("development", "queue", root)
        (batch / "guide_v0_2.md").write_text("Guide")
        with (root / "data/processed/news_documents.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "corpus changed"):
            protected_families(root)

    def test_pending_reservation_blocks_new_development(self):
        root, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "Incomplete evaluation reservation"):
            prepare("development", "queue", root)


if __name__ == "__main__":
    unittest.main()
