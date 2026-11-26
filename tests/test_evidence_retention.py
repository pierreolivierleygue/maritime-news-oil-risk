"""Boundary, metric, leakage and reproduction checks for the input audit."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_evidence_retention import (
    aggregate, audit_row, choose_limit, model_input, run, stratify,
)
from scripts.prepare_modelling_dataset import text_digest


class RetentionTests(unittest.TestCase):
    def row(self):
        body = "  Alpha\tbeta\u00a0gamma\nDelta.  "
        spans = []
        for text in ("Alpha", "gamma\nDelta."):
            start = body.index(text)
            spans.append(dict(field="body_model", start=start, end=start + len(text), text=text))
        return dict(source="FT", document_id="example", record_version_hash="v1",
                    publication_date="2020-01-01", modelling_role="primary",
                    nlp_training_eligible=True, split_group_id="group", title="gamma Delta.",
                    labels=dict(relevant=1, threat=None, disruption=0, deescalation=0),
                    body_model=body, body_model_sha256=text_digest(body), evidence_spans_model=spans,
                    annotation={"rationale": "SECRET ANNOTATION SHOULD NOT BE AN INPUT"})

    def test_unicode_word_boundaries_preserve_original_characters(self):
        row = self.row()
        result = model_input(row["title"], row["body_model"], 2)
        self.assertEqual(result["input_text"], "gamma Delta.\n\n  Alpha\tbeta")
        self.assertEqual(result["body_words_total"], 4)
        self.assertEqual(result["body_words_retained"], 2)
        self.assertEqual(result["body_start_in_input"], len(row["title"]) + 2)
        self.assertNotIn("SECRET", result["input_text"])

    def test_full_body_and_short_body_keep_trailing_whitespace(self):
        row = self.row()
        full = model_input(row["title"], row["body_model"], None)
        self.assertEqual(full, model_input(row["title"], row["body_model"], 4))
        self.assertEqual(full, model_input(row["title"], row["body_model"], 128))
        self.assertEqual(full["input_text"], row["title"] + "\n\n" + row["body_model"])
        self.assertEqual(full["input_sha256"], text_digest(full["input_text"]))

    def test_partial_span_is_not_counted_as_retained(self):
        row = self.row()
        saved = deepcopy(row)
        first = audit_row(row, 1)
        self.assertEqual(first["retained_span_count"], 1)
        self.assertEqual(first["spans"][0]["end"], first["body_end_exclusive"])
        partial = audit_row(row, 3)
        self.assertEqual(partial["spans"][1]["status"], "partial")
        self.assertEqual(partial["spans"][1]["retained_characters"], len("gamma"))
        self.assertEqual(partial["retained_span_count"], 1)
        self.assertTrue(partial["any_span_retained"])
        self.assertFalse(partial["all_spans_retained"])
        self.assertTrue(audit_row(row, None)["all_spans_retained"])
        self.assertEqual(row, saved)

    def test_headline_is_not_credited_as_body_evidence(self):
        row = self.row()
        inp = model_input(row["title"], row["body_model"], 0)
        self.assertEqual(inp["input_text"], row["title"])
        self.assertIsNone(inp["body_start_in_input"])
        result = audit_row(row, 0)
        self.assertFalse(result["any_span_retained"])
        self.assertEqual(result["retained_span_count"], 0)

    def test_invalid_evidence_and_body_drift_fail_closed(self):
        row = self.row()
        row["body_model"] += " changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit_row(row, None)
        row = self.row()
        row["evidence_spans_model"][0]["start"] += 1
        with self.assertRaisesRegex(ValueError, "offset/text mismatch"):
            audit_row(row, None)
        row["evidence_spans_model"] = []
        with self.assertRaisesRegex(ValueError, "Missing evidence"):
            audit_row(row, None)

    def test_future_protected_and_excluded_records_cannot_enter_audit(self):
        for key, value in (("publication_date", "2023-01-01"),
                           ("nlp_training_eligible", False), ("modelling_role", "excluded")):
            with self.subTest(key=key):
                row = self.row()
                row[key] = value
                with self.assertRaisesRegex(ValueError, "Ineligible record"):
                    audit_row(row, 128)

    def test_selection_uses_shortest_complete_prefix_or_full_fallback(self):
        row = self.row()
        limits = [0, 1, 3, 4, 8, None]
        self.assertEqual(choose_limit([audit_row(row, n) for n in limits], limits), 4)
        limits = [0, 1, 3, None]
        self.assertIsNone(choose_limit([audit_row(row, n) for n in limits], limits))
        with self.assertRaisesRegex(ValueError, "Even full body"):
            choose_limit([], limits)

    def test_strata_and_component_counts_do_not_claim_target_specific_evidence(self):
        rows = [audit_row(self.row(), 3), audit_row(self.row(), None)]
        result = aggregate(rows)
        self.assertEqual(result["articles_all_spans_retained"], 1)
        self.assertEqual(result["retained_spans"], 3)
        self.assertEqual(result["spans"], 4)
        self.assertEqual(result["partial_spans"], 1)
        self.assertEqual(result["related_components"], 1)
        self.assertEqual(result["components_with_all_articles_retaining_all_spans"], 0)
        strata = stratify(rows, "group")
        self.assertEqual(strata["positive_label_articles"]["relevant"]["articles"], 2)
        self.assertEqual(strata["positive_label_articles"]["threat"]["articles"], 0)
        self.assertIsNone(strata["positive_label_articles"]["threat"]["span_retention"])
        self.assertEqual(strata["without_largest_primary_component"]["articles"], 0)

    def test_invalid_limits_and_empty_text_are_rejected(self):
        for limit in (-1, True, 1.5, "256"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                model_input("Title", "Body", limit)
        for title, body in (("", "Body"), ("Title", " ")):
            with self.assertRaises(ValueError):
                model_input(title, body, None)

    def test_frozen_audit_reproduces_refuses_overwrite_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.txt"
            input_path.write_text("original")
            artifacts = {"selected_rule.json": {"body_word_limit": None},
                         "report.md": "Report\n", "inputs_primary.jsonl": [{"input_text": "Title\n\nBody"}]}
            def assemble(root):
                return artifacts, {"input.txt": text_digest(input_path.read_text())}
            with patch("scripts.audit_evidence_retention.assemble", side_effect=assemble):
                run(root)
                self.assertTrue(run(root, validate_only=True)["validated"])
                with self.assertRaisesRegex(ValueError, "Existing audit"):
                    run(root)
                report = root / "data/audit/nlp_input_v1/report.md"
                report.write_text("Changed\n")
                with self.assertRaisesRegex(ValueError, "drift"):
                    run(root, validate_only=True)
                report.write_text("Report\n")
                input_path.write_text("different input")
                with self.assertRaisesRegex(ValueError, "drift"):
                    run(root, validate_only=True)


if __name__ == "__main__":
    unittest.main()
