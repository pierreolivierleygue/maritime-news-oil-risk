"""Chronological M0/M1 validation and safe model snapshots; no final-test scoring."""
import argparse
from datetime import date, datetime, time, timedelta, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import tempfile

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from .prepare_news_review import ROOT, digest, read, write_rows
    from .prepare_modelling_dataset import text_digest
    from .reserve_news_evaluation import save_json
    from .validate_evaluation_annotations import verify_hashes
except ImportError:
    from prepare_news_review import ROOT, digest, read, write_rows
    from prepare_modelling_dataset import text_digest
    from reserve_news_evaluation import save_json
    from validate_evaluation_annotations import verify_hashes

POLICY = "config/financial_baselines_v1.json"
FEATURES = ["ovx_latest", "ovx_trailing_four_week_mean"]
WEEK = timedelta(days=7)


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Naive forecast/maturity timestamp")
    return result.astimezone(timezone.utc)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_rows(training, validation, policy):
    if policy["features"] != FEATURES or not policy["alpha_grid"] or any(
            not finite(a) or a <= 0 for a in policy["alpha_grid"]) or len(set(policy["alpha_grid"])) != len(policy["alpha_grid"]):
        raise ValueError("Invalid baseline feature set or regularisation grid")
    seen = set()
    for split, rows in (("training", training), ("validation", validation)):
        first, last = policy[f"{split}_dates"]
        if len(rows) != policy[f"expected_{split}_rows"]:
            raise ValueError("Unexpected baseline population")
        previous = None
        for r in rows:
            start, end = date.fromisoformat(r["target_week_start"]), date.fromisoformat(r["target_week_end"])
            cutoff, mature = utc(r["cutoff_exclusive_utc"]), utc(r["target_mature_after_utc"])
            feature_week = date.fromisoformat(r["feature_week_start"])
            if (r["eligible"] is not True or r["split"] != split or r["exclusion_reasons"]
                    or r["timing"] != policy["timing"] or r["view"] != policy["view"]
                    or not first <= start.isoformat() <= end.isoformat() <= last
                    or end.isoformat() >= "2024-01-01"):
                raise ValueError("Ineligible, final-test or wrong-split baseline row")
            if (start.weekday() != 0 or end != start + timedelta(days=4) or feature_week != start - WEEK
                    or cutoff != datetime.combine(start, time(), timezone.utc)
                    or mature != datetime.combine(end + timedelta(days=1), time(), timezone.utc)):
                raise ValueError("Inconsistent target/cutoff/maturity dates")
            expected_history = [(feature_week - i * WEEK).isoformat() for i in range(4)]
            if (r["ovx_history_weeks"] != expected_history or len(r["ovx_history_observations"]) != 4
                    or any(type(n) is not int or not 3 <= n <= 5 for n in r["ovx_history_observations"])
                    or type(r["target_observations"]) is not int or not 3 <= r["target_observations"] <= 5):
                raise ValueError("Wrong or insufficient market history/target observations")
            if any(not finite(r["features"][f]) or r["features"][f] <= 0 for f in FEATURES) or not finite(
                    r["target_ovx_mean"]) or r["target_ovx_mean"] <= 0:
                raise ValueError("Non-finite or nonpositive market input/target")
            if start in seen or (previous is not None and start <= previous):
                raise ValueError("Repeated or unsorted target weeks")
            seen.add(start)
            previous = start
    if not training or not validation or training[-1]["target_week_end"] >= validation[0]["target_week_start"]:
        raise ValueError("Training/validation chronology overlaps")


def matured_rows(pool, cutoff):
    cutoff = utc(cutoff)
    rows = [r for r in pool if utc(r["target_mature_after_utc"]) < cutoff
            and date.fromisoformat(r["target_week_end"]) < cutoff.date()]
    return sorted(rows, key=lambda r: r["target_week_start"])


def fit_ridge(rows, alpha, policy):
    if len(rows) < policy["minimum_training_rows"]:
        raise ValueError("Insufficient matured training rows")
    x = np.array([[r["features"][f] for f in policy["features"]] for r in rows], dtype=float)
    y = np.array([r["target_ovx_mean"] for r in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite regression training data")
    scaler = StandardScaler()
    with threadpool_limits(limits=1):
        transformed = scaler.fit_transform(x)
        model = Ridge(alpha=alpha, **policy["ridge"]).fit(transformed, y)
    raw_coef = model.coef_ / scaler.scale_
    raw_intercept = model.intercept_ - np.dot(raw_coef, scaler.mean_)
    payload = [{"target_week_start": r["target_week_start"],
                "features": [r["features"][f] for f in policy["features"]],
                "target": r["target_ovx_mean"]} for r in rows]
    return {"model": "M1", "alpha": alpha, "features": policy["features"],
            "training_rows": len(rows), "training_first_target": rows[0]["target_week_start"],
            "training_last_target_end": rows[-1]["target_week_end"],
            "training_data_sha256": text_digest(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "scaler_variance": scaler.var_.tolist(), "standardized_coef": model.coef_.tolist(),
            "intercept": float(model.intercept_), "original_unit_coef": raw_coef.tolist(),
            "original_unit_intercept": float(raw_intercept)}


def predict_saved(bundle, row):
    x = np.array([row["features"][f] for f in bundle["features"]], dtype=float)
    mean, scale, coef = (np.asarray(bundle[k], dtype=float) for k in (
        "scaler_mean", "scaler_scale", "standardized_coef"))
    if (any(a.shape != x.shape for a in (mean, scale, coef)) or np.any(scale <= 0)
            or not all(np.isfinite(a).all() for a in (x, mean, scale, coef))):
        raise ValueError("Invalid frozen regression parameters/input")
    result = float(np.dot((x - mean) / scale, coef) + bundle["intercept"])
    if not math.isfinite(result):
        raise ValueError("Non-finite prediction")
    return result


def walk_forward(training, validation, policy):
    validate_rows(training, validation, policy)
    forecasts, audit = [], []
    pool = training + validation
    for row in validation:
        earlier = matured_rows(pool, row["cutoff_exclusive_utc"])
        if any(r["target_week_start"] >= row["target_week_start"] for r in earlier):
            raise ValueError("Current/future target in chronological fit")
        metadata = {k: row[k] for k in ("feature_week_start", "cutoff_exclusive_utc",
                                       "target_week_start", "target_week_end")}
        forecasts.append({**metadata, "model": "M0", "alpha": None,
                          "prediction": row["features"]["ovx_latest"]})
        fits = []
        for alpha in policy["alpha_grid"]:
            bundle = fit_ridge(earlier, alpha, policy)
            forecasts.append({**metadata, "model": "M1", "alpha": alpha,
                              "prediction": predict_saved(bundle, row)})
            fits.append(bundle)
        audit.append({**metadata, "training_targets": [r["target_week_start"] for r in earlier],
                      "latest_training_maturity": max(r["target_mature_after_utc"] for r in earlier),
                      "forecast_features": {f: row["features"][f] for f in policy["features"]}, "fits": fits})
    return forecasts, audit


def metrics(rows):
    if not rows:
        return {"weeks": 0, "mae": None, "rmse": None, "mean_error": None}
    errors = np.array([r["prediction"] - r["actual"] for r in rows])
    if not np.isfinite(errors).all():
        raise ValueError("Non-finite forecast error")
    return {"weeks": len(rows), "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))), "mean_error": float(np.mean(errors))}


def score_validation(forecasts, validation, policy):
    truth = {r["target_week_start"]: r for r in validation}
    expected = {(model, alpha, key) for model, alpha in [("M0", None), *[("M1", a) for a in policy["alpha_grid"]]]
                for key in truth}
    actual = [(r["model"], r["alpha"], r["target_week_start"]) for r in forecasts]
    if len(truth) != len(validation) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, repeated or mismatched validation forecasts")
    scored = [{**r, "actual": truth[r["target_week_start"]]["target_ovx_mean"]} for r in forecasts]
    candidates = []
    for alpha in policy["alpha_grid"]:
        candidates.append({"alpha": alpha, **metrics([r for r in scored if r["model"] == "M1" and r["alpha"] == alpha])})
    selected = min(candidates, key=lambda r: (r["mae"], -r["alpha"]))["alpha"]
    chosen = [r for r in scored if r["model"] == "M0" or r["alpha"] == selected]
    results = {model: metrics([r for r in chosen if r["model"] == model]) for model in ("M0", "M1")}
    baseline = results["M0"]["mae"]
    comparison = {"mae_difference_M1_minus_M0": results["M1"]["mae"] - baseline,
                  "rmse_difference_M1_minus_M0": results["M1"]["rmse"] - results["M0"]["rmse"],
                  "relative_mae_reduction_percent": 100 * (baseline - results["M1"]["mae"]) / baseline if baseline else None}
    quarter = lambda r: (date.fromisoformat(r["target_week_start"]).month - 1) // 3 + 1
    quarters = {str(q): {model: metrics([r for r in chosen if r["model"] == model and quarter(r) == q])
                         for model in results} for q in range(1, 5)}
    return {"version": policy["version"], "stage": "2023_validation_selection", "selected_alpha_M1": selected,
            "models": results, "comparison": comparison, "candidates": candidates, "quarters": quarters,
            "final_test_scored": False, "news_models_evaluated": False, "limitations": policy["limitations"]}, scored, chosen


def load_inputs(root, policy):
    dataset = root / policy["dataset_directory"]
    manifest_path = dataset / "manifest.json"
    if digest(manifest_path) != policy["dataset_manifest_sha256"]:
        raise ValueError("Frozen weekly dataset manifest changed")
    manifest = json.loads(manifest_path.read_text())
    verify_hashes(root, manifest["input_sha256"])
    verify_hashes(dataset, manifest["artifact_sha256"])
    # Test/all-period files are hashed for integrity, never parsed or scored here.
    training_path, validation_path = dataset / "primary_training.jsonl", dataset / "primary_validation.jsonl"
    training, validation = read(training_path), read(validation_path)
    dependencies = {str(p.relative_to(root)): digest(p) for p in (manifest_path, training_path, validation_path,
        root / POLICY, root / "scripts/financial_baselines.py", root / "scripts/prepare_news_review.py",
        root / "scripts/prepare_modelling_dataset.py", root / "scripts/reserve_news_evaluation.py",
        root / "scripts/validate_evaluation_annotations.py")}
    return training, validation, dependencies


def compute(stage, root, policy):
    training, validation, dependencies = load_inputs(root, policy)
    forecasts, audit = walk_forward(training, validation, policy)
    summary, scored, chosen = score_validation(forecasts, validation, policy)
    cutoff = "2024-01-01T00:00:00+00:00"
    final_rows = matured_rows(training + validation, cutoff)
    final = fit_ridge(final_rows, summary["selected_alpha_M1"], policy)
    save_json(stage / "final_model.json", {"as_of_cutoff_utc": cutoff, **final})
    write_rows(stage / "candidate_predictions.jsonl", scored)
    write_rows(stage / "selected_predictions.jsonl", chosen)
    save_json(stage / "fit_audit.json", audit)
    save_json(stage / "summary.json", summary)
    save_json(stage / "protocol.json", policy)
    save_json(stage / "selection.json", {"model": "M1", "alpha": summary["selected_alpha_M1"],
        "selected_using": "2023_validation_MAE", "features": policy["features"],
        "dataset_manifest_sha256": policy["dataset_manifest_sha256"], "final_test_scored": False,
        "final_training_targets": [r["target_week_start"] for r in final_rows]})
    errors = [{**r, "error": r["prediction"] - r["actual"], "absolute_error": abs(r["prediction"] - r["actual"])}
              for r in chosen]
    write_rows(stage / "validation_errors.jsonl", sorted(errors, key=lambda r: (-r["absolute_error"], r["model"], r["target_week_start"])))
    (stage / "report.md").write_text(report(summary), encoding="utf-8")
    return summary, dependencies


def report(summary):
    lines = ["# Financial baseline validation", "", "2023 validation and regularisation selection; final test not scored.", "",
             "| Model | Weeks | MAE (OVX points) | RMSE (OVX points) |", "|---|---:|---:|---:|"]
    for name, result in summary["models"].items():
        lines.append(f"| {name} | {result['weeks']} | {result['mae']:.4f} | {result['rmse']:.4f} |")
    lines += ["", f"Selected M1 ridge alpha: {summary['selected_alpha_M1']}.",
              f"M1 minus M0 MAE: {summary['comparison']['mae_difference_M1_minus_M0']:.4f} OVX points.", "",
              *summary["limitations"], ""]
    return "\n".join(lines)


def run(root=ROOT, validate_only=False):
    policy = json.loads((root / POLICY).read_text())
    output = root / policy["output_directory"]
    if not output.resolve().is_relative_to((root / "data/experiments").resolve()):
        raise ValueError("Unsafe output directory")
    if output.exists() and not validate_only:
        raise ValueError("Baseline experiment exists; use --validate for exact reproduction")
    saved = None
    if validate_only:
        saved = json.loads((output / "manifest.json").read_text())
        verify_hashes(root, saved["input_sha256"])
        verify_hashes(output, saved["artifact_sha256"])
    policy_hash = digest(root / POLICY)
    runtime = {"python": platform.python_version(), "machine": platform.machine(),
               "packages": {k: version(k) for k in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".financial_baselines_", dir=output.parent) as temporary:
        stage = Path(temporary)
        summary, inputs = compute(stage, root, policy)
        outputs = {p.name: digest(p) for p in sorted(stage.iterdir()) if p.is_file()}
        verify_hashes(root, inputs)
        if digest(root / POLICY) != policy_hash:
            raise ValueError("Baseline policy changed during computation")
        if validate_only:
            if (saved["input_sha256"] != inputs or saved["artifact_sha256"] != outputs
                    or saved["runtime"] != runtime or saved["schema_version"] != 1):
                raise ValueError("Baseline experiment does not reproduce exactly")
        else:
            save_json(stage / "manifest.json", {"schema_version": 1, "status": "completed_validation_only",
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
