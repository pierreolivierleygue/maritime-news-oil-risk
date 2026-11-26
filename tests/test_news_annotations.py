"""Synthetic checks for saved-label loss, evaluation leakage and evidence drift."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_news_review import digest, prepare, read, write_rows
from scripts.validate_news_annotations import validate_batch


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.processed = self.root / "data/processed"
        self.processed.mkdir(parents=True)
        (self.root / "ANNOTATION_GUIDE.md").write_text("Synthetic guide")
        self.document = dict(source="Test", document_id="1", record_version_hash="v1",
                             version_hashes=["v1"], body_hash="body", body_clean="Ships face attacks near Hormuz.",
                             title="Shipping", publication_date="2020-01-01", candidate_relevant=True,
                             exact_group_id="g1", near_family_id="f1")
        write_rows(self.processed / "news_documents.jsonl", [self.document])
        self.family = dict(near_family_id="f1", exact_group_ids=["g1"], publication_dates=["2020-01-01"])
        write_rows(self.processed / "news_review_families.jsonl", [self.family])
        write_rows(self.processed / "news_screening_review_queue.jsonl", [self.document])

    def reviewed(self):
        batch = prepare("fixture", "queue", self.root)
        a = read(batch / "annotations.jsonl")[0]
        a.update(status="ai_annotated", annotator_type="ai", annotator="Synthetic AI",
                 guide_version="0.1",
                 annotated_at="2026-11-12T00:00:00+00:00", relevant=1, threat=1,
                 disruption=None, deescalation=0, rationale="No operational consequence is specified.",
                 evidence_excerpts=["attacks"],
                 evidence_spans=[dict(field="body_clean", start=11, end=18, text="attacks")])
        write_rows(batch / "annotations.jsonl", [a])
        write_rows(batch / "annotations.ai_initial.jsonl", [a])
        (batch / "review_summary.json").write_text(json.dumps(
            {"annotation_sha256": digest(batch / "annotations.ai_initial.jsonl")}))
        return batch

    def test_reviewed_null_is_preserved_and_overwrite_is_refused(self):
        batch = self.reviewed()
        before = (batch / "annotations.jsonl").read_bytes()
        self.assertIsNone(validate_batch(batch)[0]["disruption"])
        with self.assertRaisesRegex(ValueError, "Existing batch"):
            prepare("fixture", "queue", self.root)
        self.assertEqual((batch / "annotations.jsonl").read_bytes(), before)

    def test_later_member_rejects_whole_family_before_writing(self):
        self.family["publication_dates"].append("2024-01-01")
        write_rows(self.processed / "news_review_families.jsonl", [self.family])
        with self.assertRaisesRegex(ValueError, "outside the development"):
            prepare("fixture", "queue", self.root)
        self.assertFalse((self.root / "data/annotations/fixture").exists())

    def test_changed_queue_version_is_not_silently_replaced(self):
        write_rows(self.processed / "news_screening_review_queue.jsonl",
                   [{**self.document, "record_version_hash": "obsolete"}])
        with self.assertRaisesRegex(ValueError, "version changed"):
            prepare("fixture", "queue", self.root)

    def test_misquoted_evidence_and_invalid_label_dependencies_fail(self):
        batch = self.reviewed()
        rows = read(batch / "annotations.jsonl")
        rows[0]["evidence_spans"][0]["start"] = 10
        write_rows(batch / "annotations.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Evidence span"):
            validate_batch(batch)
        rows[0]["evidence_spans"][0]["start"] = 11
        rows[0]["relevant"] = 0
        write_rows(batch / "annotations.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Irrelevant article"):
            validate_batch(batch)

    def test_frozen_text_change_is_detected(self):
        batch = self.reviewed()
        write_rows(batch / "reviewed_texts.jsonl", [{**self.document, "body_clean": "Altered"}])
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            validate_batch(batch)

    def revised(self):
        batch = self.reviewed()
        before = read(batch / "annotations.jsonl")
        (batch / "annotations.guide_0_1.jsonl").write_bytes((batch / "annotations.jsonl").read_bytes())
        (batch / "guide_v0_2.md").write_text("Synthetic revised guide")
        after = json.loads(json.dumps(before))
        after[0].update(guide_version="0.2", disruption=0)
        write_rows(batch / "annotations.jsonl", after)
        labels = ("relevant", "threat", "disruption", "deescalation")
        revision = dict(guide_version="0.2", previous_snapshot="annotations.guide_0_1.jsonl",
                        initial_snapshot="annotations.ai_initial.jsonl", guide_snapshot="guide_v0_2.md",
                        current_annotation_file="annotations.jsonl", decisions=[{
                            "document_id": "1", "record_version_hash": "v1", "article_number": 1,
                            "before": {k: before[0][k] for k in labels},
                            "after": {k: after[0][k] for k in labels},
                            "changed_label_fields": ["disruption"],
                            "explanation": "Absence of qualifying evidence under the revised convention."}])
        for field, h in [("previous_snapshot", "previous_sha256"), ("initial_snapshot", "initial_sha256"),
                         ("guide_snapshot", "guide_sha256"), ("current_annotation_file", "current_annotation_sha256")]:
            revision[h] = digest(batch / revision[field])
        (batch / "revision_guide_0_2.json").write_text(json.dumps(revision))
        return batch

    def test_revision_preserves_initial_decision_and_detects_unlogged_edit(self):
        batch = self.revised()
        self.assertEqual(validate_batch(batch)[0]["disruption"], 0)
        self.assertIsNone(read(batch / "annotations.ai_initial.jsonl")[0]["disruption"])
        rows = read(batch / "annotations.jsonl")
        rows[0]["disruption"] = 1
        write_rows(batch / "annotations.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Revision hash mismatch"):
            validate_batch(batch)

    def test_revision_rejects_inaccurate_transition_ledger(self):
        batch = self.revised()
        p = batch / "revision_guide_0_2.json"
        revision = json.loads(p.read_text())
        revision["decisions"][0]["before"]["disruption"] = 1
        p.write_text(json.dumps(revision))
        with self.assertRaisesRegex(ValueError, "transition"):
            validate_batch(batch)

    def test_changed_decision_without_ledger_is_rejected(self):
        batch = self.reviewed()
        rows = read(batch / "annotations.jsonl")
        rows[0].update(guide_version="0.2", disruption=0)
        write_rows(batch / "annotations.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "require a revision ledger"):
            validate_batch(batch)


if __name__ == "__main__":
    unittest.main()
