"""Synthetic checks for evaluation reference integrity and target-specific nulls."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.validate_evaluation_annotations import summarize, validate_rows, verify_hashes


def fixture():
    body = "Ships paused in the Red Sea after attack threats."
    q = dict(source="Example", document_id="one", record_version_hash="version",
             body_hash=hashlib.sha256(body.encode()).hexdigest(), exact_group_id="exact",
             near_family_id="family", publication_date="2023-12-01", title="Example",
             split_group_id="component", partition="evaluation_reserved",
             nlp_training_eligible=False, evaluation_selected=True, evaluation_eligible=True,
             article_number=1, status="reserved_unlabelled", guide_version="0.2")
    freeze = dict(guide_version="0.2", guide_sha256="guide", reservation_id="test",
                  guide_frozen_at="2026-11-18T00:00:00+00:00")
    a = dict(q, status="ai_annotated", annotator="Synthetic test AI", annotator_type="ai",
             human_reviewed=False, input_scope="full_article", annotated_at="2026-11-21T00:00:00+00:00",
             guide_sha256="guide", reservation_id="test", relevant=1, threat=1, disruption=1,
             deescalation=0, location="Red Sea–Bab el-Mandeb", temporal_focus="current development",
             rationale="Explicit pause and attack threat.", evidence_excerpts=[body],
             evidence_spans=[dict(field="body_clean", start=0, end=len(body), text=body)],
             review_notes="")
    return [a], [q], [dict(q, body_clean=body)], freeze


class EvaluationAnnotationTests(unittest.TestCase):
    def test_valid_reference(self):
        validate_rows(*fixture())

    def test_missing_extra_and_reordered_identity(self):
        for mode in ("missing", "extra", "identity", "order"):
            with self.subTest(mode=mode):
                a, q, d, f = fixture()
                if mode == "missing":
                    a.clear()
                elif mode == "extra":
                    a.append(deepcopy(a[0]))
                elif mode == "identity":
                    a[0]["document_id"] = "other"
                else:
                    a[0]["article_number"] = 2
                with self.assertRaises(ValueError):
                    validate_rows(a, q, d, f)

    def test_reservation_cannot_become_training_or_labelled_queue(self):
        for target, key, value in ((0, "nlp_training_eligible", True),
                                   (1, "relevant", 0), (0, "split_group_id", "other"),
                                   (0, "evaluation_selected", False)):
            with self.subTest(key=key):
                args = fixture()
                args[target][0][key] = value
                with self.assertRaises(ValueError):
                    validate_rows(*args)

    def test_policy_and_provenance_are_frozen(self):
        for key, value in (("guide_sha256", "changed"), ("human_reviewed", True),
                           ("status", "reserved_unlabelled"), ("input_scope", "headline"),
                           ("annotated_at", "2026-11-17T00:00:00+00:00"),
                           ("annotated_at", "2026-11-21T00:00:00")):
            with self.subTest(key=key, value=value):
                args = fixture()
                args[0][0][key] = value
                with self.assertRaises(ValueError):
                    validate_rows(*args)

    def test_evidence_offsets_and_body_are_verified(self):
        for mode in ("offset", "excerpt", "body"):
            with self.subTest(mode=mode):
                a, q, d, f = fixture()
                if mode == "offset":
                    a[0]["evidence_spans"][0]["start"] = 1
                elif mode == "excerpt":
                    a[0]["evidence_excerpts"] = ["invented"]
                else:
                    d[0]["body_clean"] += " changed"
                with self.assertRaises(ValueError):
                    validate_rows(a, q, d, f)

    def test_label_types_dependencies_and_context(self):
        for key, value in (("threat", True), ("disruption", 2), ("relevant", 0),
                           ("relevant", None), ("location", None), ("temporal_focus", "tomorrow")):
            with self.subTest(key=key):
                args = fixture()
                args[0][0][key] = value
                with self.assertRaises(ValueError):
                    validate_rows(*args)

    def test_null_is_excluded_only_from_its_target(self):
        a, q, d, f = fixture()
        a[0]["disruption"] = None
        a[0]["review_notes"] = "Conflicting operational evidence."
        validate_rows(a, q, d, f)
        result = summarize(a)
        self.assertEqual(result["resolved_by_target"],
                         dict(relevant=1, threat=1, disruption=0, deescalation=1))
        self.assertEqual(result["counts"]["disruption"], dict(yes=0, no=0, unresolved=1))
        self.assertEqual(result["excluding_dominant_component"]["articles"], 0)
        a[0]["review_notes"] = ""
        with self.assertRaises(ValueError):
            validate_rows(a, q, d, f)

    def test_saved_artifact_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            path = base / "reference.jsonl"
            path.write_text("initial")
            hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
            verify_hashes(base, hashes)
            path.write_text("revised")
            with self.assertRaises(ValueError):
                verify_hashes(base, hashes)
            with self.assertRaises(ValueError):
                verify_hashes(base, {"../reference.jsonl": "hash"})


if __name__ == "__main__":
    unittest.main()
