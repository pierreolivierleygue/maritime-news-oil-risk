import copy
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from scripts import financial_baselines as b


def policy():
    p = json.loads((b.ROOT / b.POLICY).read_text())
    p.update(expected_training_rows=3, expected_validation_rows=2, minimum_training_rows=2,
             alpha_grid=[0.1, 10.0])
    return p


def row(start, split, x, y):
    day = date.fromisoformat(start)
    feature = day - timedelta(days=7)
    stamp = lambda d: datetime.combine(d, time(), timezone.utc).isoformat()
    return {"target_week_start": start, "target_week_end": (day + timedelta(days=4)).isoformat(),
        "feature_week_start": feature.isoformat(), "cutoff_exclusive_utc": stamp(day),
        "target_mature_after_utc": stamp(day + timedelta(days=5)), "eligible": True,
        "split": split, "timing": "publication", "view": "combined", "exclusion_reasons": [],
        "features": dict(zip(b.FEATURES, x)), "target_ovx_mean": y,
        "ovx_history_weeks": [(feature - i * timedelta(days=7)).isoformat() for i in range(4)],
        "ovx_history_observations": [5] * 4, "target_observations": 5}


def sample():
    train = [row("2022-12-12", "training", [10., 8.], 12.),
             row("2022-12-19", "training", [20., 15.], 21.),
             row("2022-12-26", "training", [30., 24.], 32.)]
    validation = [row("2023-01-02", "validation", [32., 26.], 33.),
                  row("2023-01-09", "validation", [34., 28.], 35.)]
    return train, validation


class FinancialBaselineTests(unittest.TestCase):
    def test_expanding_fit_uses_only_matured_earlier_targets(self):
        train, validation = sample()
        forecasts, audit = b.walk_forward(train, validation, policy())
        self.assertEqual([len(r["training_targets"]) for r in audit], [3, 4])
        self.assertEqual(audit[0]["training_targets"], [r["target_week_start"] for r in train])
        self.assertEqual(audit[1]["training_targets"][-1], "2023-01-02")
        for r in audit:
            self.assertLess(b.utc(r["latest_training_maturity"]), b.utc(r["cutoff_exclusive_utc"]))
        self.assertEqual([r["prediction"] for r in forecasts if r["model"] == "M0"], [32., 34.])

    def test_strict_maturity_excludes_current_future_and_delayed_labels(self):
        train, validation = sample()
        train[0]["target_mature_after_utc"] = validation[0]["cutoff_exclusive_utc"]
        train[1]["target_mature_after_utc"] = "2023-01-03T00:00:00+00:00"
        matured = b.matured_rows(train + validation, validation[0]["cutoff_exclusive_utc"])
        self.assertEqual([r["target_week_start"] for r in matured], ["2022-12-26"])

    def test_scaling_is_fitted_only_on_the_current_training_window(self):
        train, validation = sample()
        validation[0]["features"]["ovx_latest"] = 100000.
        _, audit = b.walk_forward(train, validation, policy())
        x = np.array([[r["features"][f] for f in b.FEATURES] for r in train])
        np.testing.assert_allclose(audit[0]["fits"][0]["scaler_mean"], x.mean(axis=0))
        np.testing.assert_allclose(audit[0]["fits"][0]["scaler_variance"], x.var(axis=0))
        self.assertGreater(audit[1]["fits"][0]["scaler_mean"][0], 20000)

    def test_current_and_future_target_perturbation_cannot_change_earlier_predictions(self):
        train, validation = sample()
        original, audits = b.walk_forward(train, validation, policy())
        changed = copy.deepcopy(validation)
        changed[1]["target_ovx_mean"] = 99999.
        forecasts, other_audit = b.walk_forward(train, changed, policy())
        self.assertEqual(original, forecasts)
        self.assertEqual(audits, other_audit)
        changed[0]["target_ovx_mean"] = 88888.
        forecasts, _ = b.walk_forward(train, changed, policy())
        self.assertEqual([r for r in original if r["target_week_start"] == "2023-01-02"],
                         [r for r in forecasts if r["target_week_start"] == "2023-01-02"])
        self.assertNotEqual([r for r in original if r["model"] == "M1" and r["target_week_start"] == "2023-01-09"],
                            [r for r in forecasts if r["model"] == "M1" and r["target_week_start"] == "2023-01-09"])

    def test_news_fields_do_not_enter_market_baselines(self):
        train, validation = sample()
        expected = b.walk_forward(train, validation, policy())
        for r in train + validation:
            r["features"].update(news_volume=9999, threat_share=1, disruption_share=1)
        self.assertEqual(expected, b.walk_forward(train, validation, policy()))

    def test_safe_json_prediction_matches_sklearn_and_original_unit_equation(self):
        train, validation = sample()
        bundle = json.loads(json.dumps(b.fit_ridge(train, 0.1, policy())))
        x = np.array([[r["features"][f] for f in b.FEATURES] for r in train])
        scaler = StandardScaler()
        model = Ridge(alpha=0.1, solver="svd").fit(scaler.fit_transform(x), [r["target_ovx_mean"] for r in train])
        test = np.array([validation[0]["features"][f] for f in b.FEATURES])
        expected = model.predict(scaler.transform([test]))[0]
        with patch.object(Ridge, "fit", side_effect=AssertionError("No refitting")), patch.object(StandardScaler, "fit", side_effect=AssertionError("No rescaling")):
            actual = b.predict_saved(bundle, validation[0])
        self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(actual, np.dot(test, bundle["original_unit_coef"]) + bundle["original_unit_intercept"])
        bundle["scaler_scale"][0] = 0
        with self.assertRaisesRegex(ValueError, "Invalid frozen"):
            b.predict_saved(bundle, validation[0])

    def test_constant_predictor_and_minimum_training_rows(self):
        train, validation = sample()
        for r in train:
            r["features"][b.FEATURES[1]] = 7.
        bundle = b.fit_ridge(train, 1., policy())
        self.assertEqual(bundle["scaler_scale"][1], 1.)
        self.assertTrue(np.isfinite(b.predict_saved(bundle, validation[0])))
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            b.fit_ridge(train[:1], 1., policy())

    def test_final_test_wrong_dates_duplicates_and_nonfinite_data_rejected(self):
        for field, value in [("split", "test"), ("eligible", False), ("target_week_end", "2024-01-01"),
                             ("target_mature_after_utc", "2023-01-02T00:00:00+00:00"),
                             ("target_ovx_mean", float("nan")), ("target_observations", 2)]:
            train, validation = sample(); validation[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                b.validate_rows(train, validation, policy())
        train, validation = sample()
        validation[1] = copy.deepcopy(validation[0])
        with self.assertRaisesRegex(ValueError, "Repeated"):
            b.validate_rows(train, validation, policy())
        with self.assertRaisesRegex(ValueError, "Naive"):
            b.utc("2023-01-02T00:00:00")

    def test_metrics_match_independent_sklearn_calculation(self):
        rows = [{"actual":a, "prediction":p} for a,p in [(10,12),(15,14),(20,25)]]
        result = b.metrics(rows)
        self.assertAlmostEqual(result["mae"], mean_absolute_error([10,15,20],[12,14,25]))
        self.assertAlmostEqual(result["rmse"], mean_squared_error([10,15,20],[12,14,25]) ** 0.5)
        self.assertIsNone(b.metrics([])["mae"])

    def test_selection_ties_prefer_stronger_regularisation_and_keep_persistence(self):
        _, validation = sample()
        forecasts = [{"model":m, "alpha":a, "target_week_start":r["target_week_start"],
                      "prediction":r["target_ovx_mean"]}
                     for m,a in [("M0",None),("M1",0.1),("M1",10.)] for r in validation]
        summary, _, selected = b.score_validation(forecasts, validation, policy())
        self.assertEqual(summary["selected_alpha_M1"], 10.)
        self.assertIsNone(summary["comparison"]["relative_mae_reduction_percent"])
        self.assertEqual({r["model"] for r in selected}, {"M0","M1"})
        self.assertFalse(summary["final_test_scored"])
        self.assertIsNone(summary["quarters"]["2"]["M1"]["mae"])
        for bad in [forecasts[:-1], forecasts + forecasts[:1]]:
            with self.assertRaisesRegex(ValueError, "forecasts"):
                b.score_validation(bad, validation, policy())

    def test_loader_never_parses_final_test_or_all_period_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = policy(); folder = root / p["dataset_directory"]; folder.mkdir(parents=True)
            train, validation = sample()
            b.write_rows(folder / "primary_training.jsonl", train)
            b.write_rows(folder / "primary_validation.jsonl", validation)
            (folder / "primary_test.jsonl").write_text("not parseable JSON: protected final test")
            b.save_json(folder / "manifest.json", {"input_sha256": {}, "artifact_sha256": {
                name:b.digest(folder/name) for name in ["primary_training.jsonl","primary_validation.jsonl","primary_test.jsonl"]}})
            p["dataset_manifest_sha256"] = b.digest(folder / "manifest.json")
            for name in [b.POLICY,"scripts/financial_baselines.py","scripts/prepare_news_review.py",
                         "scripts/prepare_modelling_dataset.py","scripts/reserve_news_evaluation.py","scripts/validate_evaluation_annotations.py"]:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("fixture")
            with patch.object(b,"read",wraps=b.read) as reads:
                actual_train, actual_validation, _ = b.load_inputs(root,p)
            self.assertEqual(actual_train,train);self.assertEqual(actual_validation,validation)
            self.assertEqual([c.args[0].name for c in reads.call_args_list],["primary_training.jsonl","primary_validation.jsonl"])

    def test_final_snapshot_uses_only_all_matured_training_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp);train,validation=sample()
            with patch.object(b,"load_inputs",return_value=(train,validation,{})):
                b.compute(stage,stage,policy())
            final=json.loads((stage/"final_model.json").read_text())
            self.assertEqual(final["training_rows"],5)
            self.assertEqual(final["training_last_target_end"],"2023-01-13")
            self.assertEqual(final["as_of_cutoff_utc"],"2024-01-01T00:00:00+00:00")

    def test_snapshot_refuses_overwrite_reproduces_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"config").mkdir();p=policy()
            (root/b.POLICY).write_text(json.dumps(p))
            def compute(stage,root,policy):
                b.save_json(stage/"summary.json",{"complete":True})
                return {"complete":True},{b.POLICY:b.digest(root/b.POLICY)}
            with patch.object(b,"compute",side_effect=compute):
                self.assertEqual(b.run(root),b.run(root,True))
                with self.assertRaisesRegex(ValueError,"exists"):b.run(root)
                (root/p["output_directory"]/"summary.json").write_text("changed")
                with self.assertRaisesRegex(ValueError,"hash mismatch"):b.run(root,True)


if __name__ == "__main__":
    unittest.main()
