"""Leakage, exact-version, evidence and reproducibility checks for development data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_modelling_dataset import (
    ARTIFACTS, prepare_row, run, text_digest, unique_index,
)


class ModellingDatasetTests(unittest.TestCase):
    def fixture(self):
        body = "A tanker remains detained. Sujet ERIC: Shipping; Security"
        document = dict(source="FT", document_id="doc", title="Tanker detained",
                        publication_date="2021-06-01", body_clean=body,
                        body_hash=text_digest(body), record_version_hash="v1",
                        version_hashes=["v1"], near_family_id="family", exact_group_id="exact",
                        input_file="data/processed/source.jsonl", quality_flags=[])
        span = dict(field="body_clean", start=2, end=25, text=body[2:25])
        annotation = {k: document[k] for k in (
            "source", "document_id", "title", "publication_date", "body_hash",
            "record_version_hash", "near_family_id", "exact_group_id")}
        annotation.update(batch_id="news_dev_001", article_number=1, relevant=1,
                          threat=None, disruption=1, deescalation=0, status="ai_annotated",
                          annotator_type="ai", annotator="test", annotated_at="2026-11-22",
                          human_reviewed=False, guide_version="0.2", input_scope="full_article",
                          rationale="Synthetic evidence", evidence_spans=[span],
                          evidence_excerpts=[span["text"]])
        assignment = {**document, "split_group_id": "component", "partition": "development_linked",
                      "nlp_training_eligible": True, "evaluation_selected": False,
                      "evaluation_eligible": False}
        family = dict(near_family_id="family", exact_group_ids=["exact"],
                      publication_dates=["2021-06-01"])
        policy = dict(primary_sources=["FT"], start="2019-01-01", end_exclusive="2023-01-01")
        decision = {k: document[k] for k in (
            "source", "document_id", "record_version_hash", "body_hash")}
        decision.update(action="remove_metadata_suffix", flags=["metadata_removed"],
                        reason="Verified suffix", suffix_start=body.index(" Sujet ERIC:"),
                        original_length=len(body))
        return [annotation, document, deepcopy(document), assignment, family, decision, policy]

    def test_cleanup_preserves_original_offsets_labels_and_component(self):
        args = self.fixture()
        before = deepcopy(args)
        row = prepare_row(*args)
        self.assertEqual(args, before)
        self.assertEqual(row["body_model"], "A tanker remains detained.")
        self.assertEqual(row["body_clean"], args[1]["body_clean"])
        self.assertEqual(row["evidence_spans_model"][0]["start"], 2)
        self.assertEqual(row["split_group_id"], "component")
        self.assertIsNone(row["labels"]["threat"])
        self.assertFalse(row["target_eligible"]["threat"])
        self.assertTrue(row["target_eligible"]["disruption"])

    def test_protected_and_future_families_fail_closed(self):
        for field, value in (("nlp_training_eligible", False),
                             ("evaluation_selected", True),
                             ("partition", "evaluation_reserved")):
            with self.subTest(field=field):
                args = self.fixture()
                args[3][field] = value
                with self.assertRaisesRegex(ValueError, "Protected"):
                    prepare_row(*args)
        args = self.fixture()
        args[4]["publication_dates"].append("2023-01-01")
        with self.assertRaisesRegex(ValueError, "outside development"):
            prepare_row(*args)

    def test_same_document_id_does_not_allow_wrong_source_version(self):
        args = self.fixture()
        args[2]["record_version_hash"] = "another-version"
        with self.assertRaisesRegex(ValueError, "Reviewed version"):
            prepare_row(*args)
        args = self.fixture()
        args[0]["source"] = "Guardian"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            prepare_row(*args)

    def test_metadata_representative_can_differ_only_with_identical_reviewed_text(self):
        args = self.fixture()
        args[1].update(record_version_hash="metadata-version", version_hashes=["v1", "metadata-version"])
        args[3]["record_version_hash"] = "metadata-version"
        self.assertEqual(prepare_row(*args)["record_version_hash"], "v1")
        args[2]["body_clean"] = "Other version text"
        with self.assertRaisesRegex(ValueError, "source version mismatch"):
            prepare_row(*args)

    def test_bad_offsets_and_cleanup_touching_evidence_fail(self):
        args = self.fixture()
        args[0]["evidence_spans"][0]["start"] += 1
        with self.assertRaisesRegex(ValueError, "Evidence does not match"):
            prepare_row(*args)
        args = self.fixture()
        body = args[1]["body_clean"]
        start = body.index("Security")
        span = dict(field="body_clean", start=start, end=len(body), text=body[start:])
        args[0].update(evidence_spans=[span], evidence_excerpts=[span["text"]])
        with self.assertRaisesRegex(ValueError, "remove reference evidence"):
            prepare_row(*args)

    def test_stale_quality_decisions_and_unverified_suffix_fail(self):
        args = self.fixture()
        args[5]["body_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "decision version mismatch"):
            prepare_row(*args)
        args = self.fixture()
        args[5]["suffix_start"] -= 1
        with self.assertRaisesRegex(ValueError, "suffix no longer matches"):
            prepare_row(*args)

    def test_genre_exclusion_and_supplementary_do_not_change_labels(self):
        args = self.fixture()
        args[5]["action"] = "exclude"
        row = prepare_row(*args)
        self.assertEqual(row["modelling_role"], "excluded")
        self.assertEqual(row["labels"]["disruption"], 1)
        self.assertFalse(any(row["target_eligible"].values()))
        args = self.fixture()
        args[5] = None
        args[6]["primary_sources"] = ["Guardian"]
        row = prepare_row(*args)
        self.assertEqual(row["modelling_role"], "supplementary")
        self.assertFalse(row["primary_training_eligible"])
        self.assertEqual(row["body_clean"], row["body_model"])

    def test_duplicate_identity_is_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "Duplicate identity"):
            unique_index([{"id": "a"}, {"id": "a"}], ("id",))

    def test_snapshot_reproduces_refuses_overwrite_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data/processed").mkdir(parents=True)
            input_path = root / "input.txt"
            input_path.write_text("initial")
            artifacts = {name: [] for name in ARTIFACTS}
            artifacts["summary.json"] = {"example": True}
            def synthetic_assemble(root):
                return artifacts, {"input.txt": text_digest(input_path.read_text())}
            with patch("scripts.prepare_modelling_dataset.assemble", side_effect=synthetic_assemble):
                run(root)
                self.assertTrue(run(root, validate_only=True)["validated"])
                with self.assertRaisesRegex(ValueError, "Existing dataset"):
                    run(root)
                artifact = root / "data/processed/nlp_dev_v1/development_primary.jsonl"
                artifact.write_text('{}\n')
                with self.assertRaisesRegex(ValueError, "drift"):
                    run(root, validate_only=True)
                artifact.write_text("")
                input_path.write_text("changed")
                with self.assertRaisesRegex(ValueError, "drift"):
                    run(root, validate_only=True)
                manifest = json.loads((artifact.parent / "manifest.json").read_text())
                self.assertEqual(manifest["input_sha256"]["input.txt"], text_digest("initial"))


if __name__ == "__main__":
    unittest.main()
