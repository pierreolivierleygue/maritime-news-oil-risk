"""Tests for a frozen inference-only evaluation and immutable reference join."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_nlp_holdout import (
    POLICY, compute, join_references, load_frozen, prepare_inputs, run, score_rows, summarize,
)
from scripts.prepare_modelling_dataset import text_digest
from scripts.prepare_news_review import digest, write_rows
from scripts.reserve_news_evaluation import save_json


class HoldoutTests(unittest.TestCase):
    def fixture(self, name="article", group="heldout", number=1):
        body = "A tanker was seized in the Red Sea."
        text = dict(source="FT", document_id=name, record_version_hash="version", body_clean=body,
                    body_hash=text_digest(body), exact_group_id=name, near_family_id=name,
                    publication_date="2023-06-01", title="Reported seizure")
        queue = {k: v for k, v in text.items() if k != "body_clean"}
        queue.update(split_group_id=group, article_number=number, status="reserved_unlabelled",
                     nlp_training_eligible=False, partition="evaluation_reserved",
                     evaluation_selected=True, evaluation_eligible=True)
        reference = {**queue, "status": "ai_annotated", "relevant": 1, "threat": None, "disruption": 1}
        prediction = {**queue, "method": "keyword", "predictions": dict(relevant=1, threat=0, disruption=0),
                      "probabilities": dict(relevant=1., threat=0., disruption=0.)}
        return queue, text, reference, prediction

    def test_exact_full_text_input_contains_no_reference_metadata(self):
        q, text, _, _ = self.fixture()
        saved = deepcopy((q, text))
        inputs = prepare_inputs([q], [text], [])
        self.assertEqual(inputs[0]["input_text"], text["title"] + "\n\n" + text["body_clean"])
        self.assertEqual(inputs[0]["body_end_exclusive"], len(text["body_clean"]))
        self.assertNotIn("labels", inputs[0])
        self.assertNotIn("relevant", inputs[0])
        self.assertEqual((q, text), saved)

    def test_changed_version_body_and_labelled_queue_are_rejected(self):
        q, text, _, _ = self.fixture()
        text["record_version_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "version/order"):
            prepare_inputs([q], [text], [])
        q, text, _, _ = self.fixture()
        text["body_clean"] += " Changed."
        with self.assertRaisesRegex(ValueError, "body hash"):
            prepare_inputs([q], [text], [])
        q, text, _, _ = self.fixture()
        q["labels"] = {"relevant": 1}
        with self.assertRaisesRegex(ValueError, "unlabelled"):
            prepare_inputs([q], [text], [])

    def test_training_identity_and_transitive_group_overlap_are_blocked(self):
        q, text, _, _ = self.fixture()
        for training in ([dict(source="FT", document_id="article", split_group_id="other")],
                         [dict(source="Other", document_id="other", split_group_id="heldout")]):
            with self.assertRaisesRegex(ValueError, "overlaps frozen training"):
                prepare_inputs([q], [text], training)

    def test_wrong_year_eligibility_duplicate_and_missing_text_fail_closed(self):
        for key, value in (("nlp_training_eligible", True), ("evaluation_selected", False),
                           ("publication_date", "2024-01-01")):
            q, text, _, _ = self.fixture()
            q[key] = value
            if key == "publication_date":
                text[key] = value
            with self.assertRaisesRegex(ValueError, "eligibility/date"):
                prepare_inputs([q], [text], [])
        q, text, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "queue/text"):
            prepare_inputs([q, q], [text], [])
        with self.assertRaisesRegex(ValueError, "queue/text"):
            prepare_inputs([q], [], [])

    def test_reference_join_is_exact_and_does_not_change_predictions(self):
        _, _, ref, pred = self.fixture()
        saved = deepcopy(pred)
        rows = join_references([pred], [ref], ["keyword"])
        self.assertEqual(pred, saved)
        self.assertIsNone(rows[0]["labels"]["threat"])
        self.assertNotIn("labels", pred)
        ref["split_group_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "component mismatch"):
            join_references([pred], [ref], ["keyword"])

    def test_missing_extra_and_duplicate_method_predictions_are_rejected(self):
        _, _, ref, pred = self.fixture()
        for predictions in ([], [pred, pred], [pred, {**pred, "method": "unexpected"}]):
            with self.assertRaises(ValueError):
                join_references(predictions, [ref], ["keyword"])
        with self.assertRaisesRegex(ValueError, "Missing or repeated"):
            join_references([pred], [ref, ref], ["keyword"])

    def test_target_specific_nulls_and_empty_metrics(self):
        _, _, ref, pred = self.fixture()
        row = join_references([pred], [ref], ["keyword"])[0]
        result = score_rows([row])
        self.assertEqual(result["targets"]["relevant"]["n"], 1)
        self.assertEqual(result["targets"]["threat"]["unresolved"], 1)
        self.assertIsNone(result["targets"]["threat"]["f1"])
        self.assertEqual(result["macro_positive_f1"], .5)
        self.assertIsNone(score_rows([])["macro_positive_f1"])

    def test_dominant_component_sensitivity_keeps_original_primary(self):
        queue, refs, predictions = [], [], []
        for i, group in enumerate(["large", "large", "small"], 1):
            q, _, ref, pred = self.fixture(str(i), group, i)
            queue.append(q); refs.append(ref)
            predictions.extend([pred, {**pred, "method": "alternative",
                                       "predictions": dict(relevant=1, threat=0, disruption=1)}])
        methods = ["keyword", "alternative"]
        scored = join_references(predictions, refs, methods)
        summary = summarize(scored, queue, dict(methods=methods, primary_method="keyword", protocol_version="test", uncertainty="dependent"))
        self.assertEqual(summary["dominant_component"]["articles"], 2)
        self.assertEqual(summary["methods"]["keyword"]["without_dominant_component"]["articles"], 1)
        self.assertEqual(summary["primary_method"], "keyword")
        self.assertEqual(summary["methods"]["alternative"]["role"], "frozen_alternative")
        self.assertFalse(summary["selection_changed"])

    def test_predictions_are_committed_before_reference_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "stage"
            stage.mkdir()
            reservation = root / "data/evaluation/test"
            (reservation / "labelling_v1").mkdir(parents=True)
            q, text, ref, _ = self.fixture()
            write_rows(reservation / "queue_snapshot.jsonl", [q])
            write_rows(reservation / "evaluation_texts.jsonl", [text])
            write_rows(reservation / "labelling_v1/annotations.jsonl", [ref])
            policy = dict(reservation_id="test", development_experiment="development", methods=["keyword"],
                          primary_method="keyword", expected_articles=1, expected_components=1,
                          expected_dominant_component_articles=1, frozen_manifest_sha256={},
                          protocol_version="test", uncertainty="test")
            for name in [POLICY, "scripts/evaluate_nlp_holdout.py", "scripts/validate_evaluation_annotations.py",
                         "data/experiments/development/selection.json"]:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
            def references_checked(*args):
                commit = json.loads((stage / "prediction_commit.json").read_text())
                self.assertFalse(commit["reference_join_performed"])
                self.assertEqual(commit["predictions_sha256"], digest(stage / "predictions.jsonl"))
                self.assertNotIn("labels", read_prediction(stage))
            coverage = [{"tokens": 1, "chunks": [{"token_start": 0, "token_end": 1}]}]
            result_pred = {**q, "method": "keyword", "probabilities": {t: 1. for t in ("relevant", "threat", "disruption")},
                           "predictions": {t: 1 for t in ("relevant", "threat", "disruption")},
                           "target_modes": {t: "keyword" for t in ("relevant", "threat", "disruption")}}
            with patch("scripts.evaluate_nlp_holdout.load_frozen", return_value=({"embedding": {}}, {"training_inputs": []}, {}, {})), \
                 patch("scripts.evaluate_nlp_holdout.validate_reservation"), \
                 patch("scripts.evaluate_nlp_holdout.encode_articles", return_value=(np.ones((1, 2)), coverage)), \
                 patch("scripts.evaluate_nlp_holdout.predict_methods", return_value=[result_pred]), \
                 patch("scripts.evaluate_nlp_holdout.validate_references", side_effect=references_checked), \
                 patch("scripts.evaluate_nlp_holdout.report_markdown", return_value="Report"):
                summary, _, _ = compute(stage, root, policy)
                self.assertFalse(summary["models_refitted"])

    def test_completed_evaluation_refuses_overwrite_and_validation_checks_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config").mkdir()
            policy = dict(output_directory="data/experiments/holdout", frozen_manifest_sha256={})
            save_json(root / POLICY, policy)
            def fake_compute(stage, root, policy):
                save_json(stage / "summary.json", {"primary_method": "keyword"})
                return {"primary_method": "keyword"}, {POLICY: digest(root / POLICY)}, {}
            with patch("scripts.evaluate_nlp_holdout.compute", side_effect=fake_compute):
                run(root)
                self.assertEqual(run(root, validate_only=True)["primary_method"], "keyword")
                with self.assertRaisesRegex(ValueError, "already recorded"):
                    run(root)
                (root / "data/experiments/holdout/summary.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    run(root, validate_only=True)

    def test_manifest_pin_fails_before_loading_any_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_frozen(root, {"frozen_manifest_sha256": {"manifest.json": text_digest("original")}})


def read_prediction(stage):
    return json.loads((stage / "predictions.jsonl").read_text().splitlines()[0])


if __name__ == "__main__":
    unittest.main()
