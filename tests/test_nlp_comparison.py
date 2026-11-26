"""Leakage, small-class fallback, token coverage and saved-model checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.compare_nlp import (
    TARGETS, candidate_key, constrain_predictions, features, fit_targets, make_folds,
    metric, predict_frozen, probabilities, run, safe_transform, tfidf_parameters,
)
from scripts.nlp_features import keyword_predictions, partition_tokens, pool_chunks
from scripts.prepare_modelling_dataset import text_digest

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config/nlp_comparison_v1.json").read_text())


def row(name, day, group, labels=None):
    return dict(source="FT", document_id=name, record_version_hash=name, split_group_id=group,
                publication_date=day, modelling_role="primary", primary_training_eligible=True,
                labels=labels or dict(relevant=0, threat=0, disruption=0))


class ComparisonTests(unittest.TestCase):
    def test_fold_purges_entire_related_component_and_future_dates(self):
        rows = [row("a", "2019-01-01", "shared"), row("b", "2019-07-01", "shared"),
                row("c", "2019-08-01", "independent"), row("d", "2020-04-01", "shared"),
                row("future", "2022-01-01", "future")]
        fold = make_folds(rows, [2020])[0]
        self.assertEqual(fold["train_indices"], [2])
        self.assertEqual(fold["purged_indices"], [0, 1])
        self.assertEqual(fold["validation_indices"], [3])

    def test_fold_rejects_ineligible_repeated_or_empty_training(self):
        rows = [row("a", "2019-01-01", "g"), row("b", "2020-01-01", "g")]
        with self.assertRaisesRegex(ValueError, "Empty purged"):
            make_folds(rows, [2020])
        rows[1]["publication_date"] = "2023-01-01"
        with self.assertRaisesRegex(ValueError, "Ineligible"):
            make_folds(rows, [2020])
        with self.assertRaisesRegex(ValueError, "Repeated"):
            make_folds([rows[0], rows[0]], [2020])

    def test_validation_only_words_never_enter_tfidf_vocabulary(self):
        texts = ["tankers sail peacefully", "naval escort protects tankers", "validationsecret blockade"]
        train, val, vectorizer = features("tfidf", texts, None, [0, 1], [2], CONFIG)
        self.assertNotIn("validationsecret", vectorizer.vocabulary_)
        self.assertNotIn("blockade", vectorizer.vocabulary_)
        self.assertEqual(val.nnz, 0)
        self.assertEqual(train.shape[0], 2)

    def test_targets_with_few_or_null_labels_fall_back_independently(self):
        rows = [row(str(i), "2019-01-01", str(i), dict(
            relevant=i % 2, threat=None if i == 1 else i % 2, disruption=i % 2)) for i in range(4)]
        matrix = np.array([[1, 0], [0, 1], [1, .1], [.1, 1]])
        config = {**CONFIG, "minimum_class_count": 2}
        saved = deepcopy(rows)
        models = fit_targets(matrix, rows, [0, 1, 2, 3], 1, config)
        self.assertEqual(models["relevant"]["mode"], "logistic")
        self.assertEqual(models["threat"]["mode"], "keyword_fallback")
        self.assertEqual(models["disruption"]["mode"], "logistic")
        probs = probabilities(matrix, models, [dict(relevant=0, threat=1, disruption=0)] * 4)
        np.testing.assert_array_equal(probs[:, 1], np.ones(4))
        self.assertEqual(rows, saved)

    def test_event_gating_never_changes_raw_probabilities(self):
        probs = np.array([[.2, .9, .8], [.8, .6, .1]])
        saved = probs.copy()
        np.testing.assert_array_equal(constrain_predictions(probs, .5), [[0, 0, 0], [1, 1, 0]])
        np.testing.assert_array_equal(probs, saved)
        with self.assertRaisesRegex(ValueError, "Invalid prediction"):
            constrain_predictions(np.array([[np.nan, 0, 1]]), .5)

    def test_metrics_nulls_and_confusion_counts_are_target_specific(self):
        result = metric([1, 1, 0, 0, None], [1, 0, 1, 0, 1])
        self.assertEqual([result[k] for k in ["n", "tp", "fp", "fn", "tn"]], [4, 1, 1, 1, 1])
        self.assertEqual(result["f1"], .5)
        self.assertIsNone(metric([None], [1])["f1"])
        self.assertEqual(metric([1], [0])["recall"], 0)

    def test_chunk_partition_covers_each_token_once_and_tail(self):
        tokens = list(range(511))
        offsets = [(i * 2, i * 2 + 1) for i in tokens]
        chunks = partition_tokens(tokens, offsets, 254)
        self.assertEqual([len(c["token_ids"]) for c in chunks], [254, 254, 3])
        self.assertEqual([t for c in chunks for t in c["token_ids"]], tokens)
        self.assertEqual(chunks[-1]["token_end"], len(tokens))
        self.assertEqual(chunks[-1]["character_end"], offsets[-1][1])
        for bad_tokens, bad_offsets, capacity in [([], [], 254), ([1], [], 254), ([1], [(0, 1)], 0)]:
            with self.assertRaises(ValueError):
                partition_tokens(bad_tokens, bad_offsets, capacity)

    def test_chunk_pooling_is_token_weighted_and_normalized(self):
        result = pool_chunks([[2, 0], [0, 7]], [3, 1])
        np.testing.assert_allclose(result, np.array([3, 1]) / np.sqrt(10), atol=1e-7)
        with self.assertRaises(ValueError):
            pool_chunks([[0, 0]], [1])

    def test_attack_only_rule_does_not_imply_operational_disruption(self):
        rule = CONFIG["keyword"]
        result = keyword_predictions("A missile attack threatens tankers in the Strait of Hormuz.", rule)
        self.assertEqual(result, dict(relevant=1, threat=1, disruption=0))
        result = keyword_predictions("A tanker was seized in the Strait of Hormuz.", rule)
        self.assertEqual(result["disruption"], 1)
        self.assertEqual(keyword_predictions("A tanker was seized in the Baltic.", rule)["relevant"], 0)
        self.assertEqual(keyword_predictions("Hormuz " + "ordinary " * 100 + "tanker seized", rule)["relevant"], 0)

    def test_saved_tfidf_and_linear_parameters_predict_without_refit(self):
        texts = ["ordinary shipping business", "threat attack tanker", "ordinary company", "threat closure"]
        rows = [row(str(i), "2019-01-01", str(i), {t: i % 2 for t in TARGETS}) for i in range(4)]
        config = {**CONFIG, "minimum_class_count": 2}
        train, _, vectorizer = features("tfidf", texts, None, list(range(4)), list(range(4)), config)
        models = fit_targets(train, rows, list(range(4)), 1, config)
        saved_vectorizer = {"vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
                            "idf": vectorizer.idf_.tolist()}
        restored = safe_transform(json.loads(json.dumps(saved_vectorizer)), texts, config)
        np.testing.assert_allclose(restored.toarray(), train.toarray())
        bundle = json.loads(json.dumps(dict(method="tfidf", targets=models, threshold=.5, vectorizer=saved_vectorizer)))
        rules = [keyword_predictions(t, config["keyword"]) for t in texts]
        expected = probabilities(train, models, rules)
        probs, predictions = predict_frozen(bundle, texts, None, config)
        np.testing.assert_allclose(probs, expected)
        np.testing.assert_array_equal(predictions, constrain_predictions(expected, .5))

    def test_selection_ties_follow_declared_complexity_and_regularization_order(self):
        def candidate(method, c, threshold):
            return dict(method=method, c=c, threshold=threshold,
                        metrics={"overall": {"macro_positive_f1": .5}})
        options = [candidate("embedding", 1, .35), candidate("tfidf", 10, .5),
                   candidate("tfidf", .1, .35), candidate("tfidf", .1, .5)]
        self.assertEqual(min(options, key=candidate_key), options[-1])
        options.append(candidate("keyword", None, None))
        self.assertEqual(min(options, key=candidate_key)["method"], "keyword")

    def test_experiment_snapshot_refuses_overwrite_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.txt"
            source.write_text("original")
            artifacts = {"summary.json": {"example": True}, "selected_predictions.jsonl": []}
            def compute(root, recompute_embeddings):
                return artifacts, {"input.txt": text_digest(source.read_text())}
            with patch("scripts.compare_nlp.compute", side_effect=compute):
                run(root)
                self.assertEqual(run(root, validate_only=True), {"example": True})
                with self.assertRaisesRegex(ValueError, "Existing experiment"):
                    run(root)
                source.write_text("changed")
                with self.assertRaisesRegex(ValueError, "drift"):
                    run(root, validate_only=True)


if __name__ == "__main__":
    unittest.main()
