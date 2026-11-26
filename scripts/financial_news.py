"""Chronological M2/M3 validation against immutable M0/M1; no final-test scoring."""
import argparse
from datetime import date, datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import tempfile

import numpy as np

try:
    from . import financial_baselines as b
except ImportError:
    import financial_baselines as b

POLICY = "config/financial_news_v1.json"
MODEL_FEATURES = {"M2": b.FEATURES + ["news_volume"],
                  "M3": b.FEATURES + ["news_volume", "threat_share", "disruption_share"]}
MODELS = ("M0", "M1", "M2", "M3")


def validate_rows(training, validation, policy):
    b.validate_rows(training, validation, policy)
    if policy["model_features"] != MODEL_FEATURES:
        raise ValueError("Unexpected news model specifications")
    for row in training + validation:
        f, count = row["features"], row["candidate_count"]
        if (type(count) is not int or count < 0
                or row["candidate_empty"] is not (count == 0)
                or any(not b.finite(f[k]) for k in MODEL_FEATURES["M3"])):
            raise ValueError("Invalid news count, empty flag or non-finite feature")
        if (not 0 <= f["news_volume"] <= math.log1p(count) + 1e-12
                or any(not 0 <= f[k] <= 1 for k in ("threat_share", "disruption_share"))
                or (count == 0 and any(f[k] != 0 for k in MODEL_FEATURES["M3"][2:]))):
            raise ValueError("Invalid news feature range or empty-week convention")


def fit_model(rows, model, alpha, policy):
    bundle = b.fit_ridge(rows, alpha, {**policy, "features": policy["model_features"][model]})
    return {**bundle, "model": model}


def walk_forward(training, validation, policy):
    validate_rows(training, validation, policy)
    forecasts, audit = [], []
    for row in validation:
        earlier = b.matured_rows(training + validation, row["cutoff_exclusive_utc"])
        if any(r["target_week_start"] >= row["target_week_start"] for r in earlier):
            raise ValueError("Current/future target in chronological news fit")
        metadata = {k: row[k] for k in ("feature_week_start", "cutoff_exclusive_utc",
                                       "target_week_start", "target_week_end")}
        fits = []
        for model in MODEL_FEATURES:
            for alpha in policy["alpha_grid"]:
                bundle = fit_model(earlier, model, alpha, policy)
                forecasts.append({**metadata, "model": model, "alpha": alpha,
                                  "prediction": b.predict_saved(bundle, row)})
                fits.append(bundle)
        audit.append({**metadata, "training_targets": [r["target_week_start"] for r in earlier],
                      "latest_training_maturity": max(r["target_mature_after_utc"] for r in earlier),
                      "forecast_features": {f: row["features"][f] for f in MODEL_FEATURES["M3"]},
                      "fits": fits})
    return forecasts, audit


def score_validation(forecasts, baselines, validation, policy):
    truth = {r["target_week_start"]: r for r in validation}
    expected = {(m, a, key) for m in MODEL_FEATURES for a in policy["alpha_grid"] for key in truth}
    keys = [(r["model"], r["alpha"], r["target_week_start"]) for r in forecasts]
    base_keys = [(r["model"], r["target_week_start"]) for r in baselines]
    if (len(truth) != len(validation) or len(keys) != len(set(keys)) or set(keys) != expected
            or len(base_keys) != len(set(base_keys))
            or set(base_keys) != {(m, key) for m in ("M0", "M1") for key in truth}):
        raise ValueError("Missing, repeated or mismatched validation forecasts")
    for row in forecasts + baselines:
        actual = truth[row["target_week_start"]]
        if not b.finite(row["prediction"]):
            raise ValueError("Non-finite validation forecast")
        for field in ("feature_week_start", "cutoff_exclusive_utc", "target_week_end"):
            if row[field] != actual[field]:
                raise ValueError("Mismatched forecast dates")
        if row["model"] in ("M0", "M1") and row["actual"] != actual["target_ovx_mean"]:
            raise ValueError("Frozen baseline actual differs from dataset")
    scored = [{**r, "actual": truth[r["target_week_start"]]["target_ovx_mean"]} for r in forecasts]
    candidates = {m: [{"alpha": a, **b.metrics([r for r in scored if r["model"] == m and r["alpha"] == a])}
                      for a in policy["alpha_grid"]] for m in MODEL_FEATURES}
    selected = {m: min(rows, key=lambda r: (r["mae"], -r["alpha"]))["alpha"] for m, rows in candidates.items()}
    chosen = sorted(baselines + [r for r in scored if r["alpha"] == selected[r["model"]]],
                    key=lambda r: (r["target_week_start"], r["model"]))
    results = {m: b.metrics([r for r in chosen if r["model"] == m]) for m in MODELS}
    comparisons = {}
    for augmented, reference in (("M3", "M2"), ("M2", "M1"), ("M1", "M0"), ("M2", "M0"), ("M3", "M0")):
        denom = results[reference]["mae"]
        comparisons[f"{augmented}_minus_{reference}"] = {
            "mae_difference": results[augmented]["mae"] - denom,
            "rmse_difference": results[augmented]["rmse"] - results[reference]["rmse"],
            "relative_mae_reduction_percent": 100 * (denom - results[augmented]["mae"]) / denom if denom else None}
    quarters = {str(q): {m: b.metrics([r for r in chosen if r["model"] == m
                    and (date.fromisoformat(r["target_week_start"]).month - 1) // 3 + 1 == q])
                    for m in MODELS} for q in range(1, 5)}
    return {"version": policy["version"], "stage": "2023_validation_selection",
            "selected_alphas": selected, "models": results, "comparisons": comparisons,
            "candidates": candidates, "quarters": quarters, "final_test_scored": False,
            "limitations": policy["limitations"]}, scored, chosen


def training_diagnostics(training):
    features = MODEL_FEATURES["M3"]
    x = np.array([[r["features"][f] for f in features] for r in training])
    centered = x - x.mean(axis=0)
    norms = np.sqrt(np.sum(centered ** 2, axis=0))
    correlation = [[float(np.dot(centered[:, i], centered[:, j]) / (norms[i] * norms[j]))
                    if norms[i] and norms[j] else None for j in range(len(features))]
                   for i in range(len(features))]
    return {"population": "initial_training_only", "rows": len(training), "features": features,
            "mean": x.mean(axis=0).tolist(), "population_std": x.std(axis=0).tolist(),
            "constant_features": [f for f, n in zip(features, norms) if n == 0],
            "pearson_correlation": correlation}


def load_inputs(root, policy):
    baseline_dir = root / policy["baseline_directory"]
    manifest_path = baseline_dir / "manifest.json"
    if b.digest(manifest_path) != policy["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline manifest changed")
    manifest = json.loads(manifest_path.read_text())
    b.verify_hashes(root, manifest["input_sha256"])
    b.verify_hashes(baseline_dir, manifest["artifact_sha256"])
    baseline_policy = json.loads((root / b.POLICY).read_text())
    for key in ("dataset_directory", "dataset_manifest_sha256", "timing", "view", "training_dates",
                "validation_dates", "expected_training_rows", "expected_validation_rows",
                "minimum_training_rows", "features", "alpha_grid", "ridge"):
        if policy[key] != baseline_policy[key]:
            raise ValueError(f"News policy differs from baseline design: {key}")
    training, validation, dependencies = b.load_inputs(root, baseline_policy)
    # Only training/validation rows and saved validation forecasts are parsed.
    predictions_path = baseline_dir / "selected_predictions.jsonl"
    baselines = b.read(predictions_path)
    dependencies.update({str(p.relative_to(root)): b.digest(p) for p in (
        root / POLICY, root / "scripts/financial_news.py", manifest_path, predictions_path)})
    return training, validation, baselines, dependencies


def compute(stage, root, policy):
    training, validation, baselines, dependencies = load_inputs(root, policy)
    forecasts, audit = walk_forward(training, validation, policy)
    summary, scored, chosen = score_validation(forecasts, baselines, validation, policy)
    cutoff = "2024-01-01T00:00:00+00:00"
    final_rows = b.matured_rows(training + validation, cutoff)
    final = {m: fit_model(final_rows, m, a, policy) for m, a in summary["selected_alphas"].items()}
    b.save_json(stage / "final_models.json", {"as_of_cutoff_utc": cutoff, "models": final})
    b.write_rows(stage / "candidate_predictions.jsonl", scored)
    b.write_rows(stage / "selected_predictions.jsonl", chosen)
    b.save_json(stage / "fit_audit.json", audit)
    b.save_json(stage / "training_diagnostics.json", training_diagnostics(training))
    b.save_json(stage / "summary.json", summary)
    b.save_json(stage / "protocol.json", policy)
    b.save_json(stage / "selection.json", {"models": {m: {"alpha": a, "features": MODEL_FEATURES[m]}
        for m, a in summary["selected_alphas"].items()}, "selected_using": "2023_validation_MAE",
        "dataset_manifest_sha256": policy["dataset_manifest_sha256"],
        "baseline_manifest_sha256": policy["baseline_manifest_sha256"], "final_test_scored": False,
        "final_training_targets": [r["target_week_start"] for r in final_rows]})
    errors = [{**r, "error": r["prediction"] - r["actual"], "absolute_error": abs(r["prediction"] - r["actual"])}
              for r in chosen]
    b.write_rows(stage / "validation_errors.jsonl", sorted(errors, key=lambda r: (-r["absolute_error"], r["model"], r["target_week_start"])))
    paired = []
    for row in validation:
        week = row["target_week_start"]
        by_model = {r["model"]: r for r in errors if r["target_week_start"] == week}
        paired.append({"target_week_start": week,
                       "absolute_error_M3_minus_M2": by_model["M3"]["absolute_error"] - by_model["M2"]["absolute_error"],
                       "squared_error_M3_minus_M2": by_model["M3"]["error"] ** 2 - by_model["M2"]["error"] ** 2})
    b.write_rows(stage / "paired_errors.jsonl", paired)
    (stage / "report.md").write_text(report(summary), encoding="utf-8")
    return summary, dependencies


def report(summary):
    lines = ["# Financial news-model validation", "", "2023 validation and regularisation selection; final test not scored.", "",
             "| Model | Weeks | MAE (OVX points) | RMSE (OVX points) |", "|---|---:|---:|---:|"]
    for name, result in summary["models"].items():
        lines.append(f"| {name} | {result['weeks']} | {result['mae']:.4f} | {result['rmse']:.4f} |")
    lines += ["", f"Selected ridge alphas: {summary['selected_alphas']}.",
              f"M3 minus M2 MAE: {summary['comparisons']['M3_minus_M2']['mae_difference']:.4f} OVX points.", "",
              *summary["limitations"], ""]
    return "\n".join(lines)


def run(root=b.ROOT, validate_only=False):
    policy = json.loads((root / POLICY).read_text())
    output = root / policy["output_directory"]
    if not output.resolve().is_relative_to((root / "data/experiments").resolve()):
        raise ValueError("Unsafe output directory")
    if output.exists() and not validate_only:
        raise ValueError("News experiment exists; use --validate for exact reproduction")
    saved = None
    if validate_only:
        saved = json.loads((output / "manifest.json").read_text())
        b.verify_hashes(root, saved["input_sha256"])
        b.verify_hashes(output, saved["artifact_sha256"])
    policy_hash = b.digest(root / POLICY)
    runtime = {"python": platform.python_version(), "machine": platform.machine(),
               "packages": {k: version(k) for k in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".financial_news_", dir=output.parent) as temporary:
        stage = Path(temporary)
        summary, inputs = compute(stage, root, policy)
        outputs = {p.name: b.digest(p) for p in sorted(stage.iterdir()) if p.is_file()}
        b.verify_hashes(root, inputs)
        if b.digest(root / POLICY) != policy_hash:
            raise ValueError("News policy changed during computation")
        if validate_only:
            if (saved["input_sha256"] != inputs or saved["artifact_sha256"] != outputs
                    or saved["runtime"] != runtime or saved["schema_version"] != 1):
                raise ValueError("News experiment does not reproduce exactly")
        else:
            b.save_json(stage / "manifest.json", {"schema_version": 1, "status": "completed_validation_only",
                "created_at": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
                "input_sha256": inputs, "artifact_sha256": outputs})
            output.mkdir(exist_ok=False)
            for path in sorted(stage.iterdir()):
                path.rename(output / path.name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(report(run(validate_only=args.validate)))
