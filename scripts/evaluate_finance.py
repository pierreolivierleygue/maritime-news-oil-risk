"""Frozen 2024–2025 financial evaluation, paired uncertainty and fixed sensitivities."""
import argparse
from datetime import date, datetime, time, timedelta, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import tempfile

import numpy as np

try:
    from . import financial_baselines as b
    from . import financial_news as n
except ImportError:
    import financial_baselines as b
    import financial_news as n

POLICY = "config/financial_evaluation_v1.json"
LOCK = "config/financial_evaluation_v1.lock.json"
MODEL_FEATURES = {"M1": b.FEATURES, **n.MODEL_FEATURES}
MODELS = ("M0", "M1", "M2", "M3")
DAY, WEEK = timedelta(days=1), timedelta(days=7)


def validate_policy(root, policy):
    lock = json.loads((root / LOCK).read_text())
    if lock["policy_sha256"] != b.digest(root / POLICY) or lock["version"] != policy["version"]:
        raise ValueError("Final evaluation policy differs from its pre-test lock")
    if (policy["model_features"] != MODEL_FEATURES or set(policy["selected_alphas"]) != set(MODEL_FEATURES)
            or any(not b.finite(a) or a <= 0 for a in policy["selected_alphas"].values())
            or policy["ridge"] != {"solver": "svd", "fit_intercept": True}):
        raise ValueError("Invalid frozen financial model specification")
    boot = policy["bootstrap"]
    if (boot["primary_block_weeks"] != math.ceil(boot["development_validation_weeks"] ** (1 / 3))
            or type(boot["replicates"]) is not int or boot["replicates"] < 1
            or not 0 < boot["confidence"] < 1):
        raise ValueError("Invalid frozen bootstrap rule")


def validate_panel(rows, panel, policy):
    counts = {split: 0 for split in policy["splits"]}
    previous = None
    for r in rows:
        start, end = date.fromisoformat(r["target_week_start"]), date.fromisoformat(r["target_week_end"])
        feature = start - WEEK
        if (r["eligible"] is not True or r["exclusion_reasons"] or r["split"] not in counts
                or r["timing"] != panel["timing"] or r["view"] != panel["view"]):
            raise ValueError("Ineligible or wrong-panel financial row")
        first, last = policy["splits"][r["split"]]
        if not first <= start.isoformat() <= end.isoformat() <= last:
            raise ValueError("Target crosses frozen split boundaries")
        if (previous is not None and start <= previous):
            raise ValueError("Repeated or unsorted financial target weeks")
        previous = start
        if (start.weekday() != 0 or end != start + 4 * DAY
                or r["feature_week_start"] != feature.isoformat()
                or b.utc(r["cutoff_exclusive_utc"]) != datetime.combine(start, time(), timezone.utc)
                or b.utc(r["target_mature_after_utc"]) != datetime.combine(end + DAY, time(), timezone.utc)):
            raise ValueError("Inconsistent target/cutoff/maturity dates")
        history = [(feature - (panel["ovx_lag_weeks"] + i) * WEEK).isoformat() for i in range(4)]
        if (r["ovx_history_weeks"] != history or len(r["ovx_history_observations"]) != 4
                or any(type(v) is not int or not 3 <= v <= 5 for v in r["ovx_history_observations"])
                or type(r["target_observations"]) is not int or not 3 <= r["target_observations"] <= 5):
            raise ValueError("Invalid lagged market history or target observations")
        f, count = r["features"], r["candidate_count"]
        if (any(not b.finite(f[k]) for k in MODEL_FEATURES["M3"])
                or any(f[k] <= 0 for k in b.FEATURES)
                or not b.finite(r["target_ovx_mean"]) or r["target_ovx_mean"] <= 0
                or type(count) is not int or count < 0 or r["candidate_empty"] is not (count == 0)):
            raise ValueError("Invalid market/news input, target or count")
        if (not 0 <= f["news_volume"] <= math.log1p(count) + 1e-12
                or any(not 0 <= f[k] <= 1 for k in ("threat_share", "disruption_share"))
                or (count == 0 and any(f[k] != 0 for k in MODEL_FEATURES["M3"][2:]))):
            raise ValueError("Invalid news range or empty-week convention")
        if not r["coverage"] or not all(c["accepted_observed_coverage"] is True for c in r["coverage"]):
            raise ValueError("Unaccepted source coverage in eligible row")
        counts[r["split"]] += 1
    if counts != panel["expected_rows"]:
        raise ValueError(f"Unexpected financial panel population: {counts}")


def available_rows(rows, cutoff, delay_days):
    if type(delay_days) is not int or delay_days < 0:
        raise ValueError("Invalid target availability delay")
    stamp = b.utc(cutoff)
    return [r for r in b.matured_rows(rows, cutoff)
            if b.utc(r["target_mature_after_utc"]) + delay_days * DAY < stamp]


def walk_forward(rows, panel, policy):
    validate_panel(rows, panel, policy)
    forecasts, audit = [], []
    for row in (r for r in rows if r["split"] == "test"):
        earlier = available_rows(rows, row["cutoff_exclusive_utc"], panel["target_availability_delay_days"])
        if any(r["target_week_start"] >= row["target_week_start"] for r in earlier):
            raise ValueError("Current/future target in final fit")
        metadata = {k: row[k] for k in ("feature_week_start", "cutoff_exclusive_utc", "target_week_start", "target_week_end")}
        forecasts.append({**metadata, "model": "M0", "alpha": None, "prediction": row["features"]["ovx_latest"]})
        fits = []
        for model, fields in policy["model_features"].items():
            bundle = b.fit_ridge(earlier, policy["selected_alphas"][model], {**policy, "features": fields})
            bundle["model"] = model
            fits.append(bundle)
            forecasts.append({**metadata, "model": model, "alpha": bundle["alpha"],
                              "prediction": b.predict_saved(bundle, row)})
        audit.append({**metadata, "training_targets": [r["target_week_start"] for r in earlier],
                      "latest_training_available_at": max((b.utc(r["target_mature_after_utc"])
                          + panel["target_availability_delay_days"] * DAY).isoformat() for r in earlier),
                      "target_availability_delay_days": panel["target_availability_delay_days"],
                      "forecast_features": {f: row["features"][f] for f in MODEL_FEATURES["M3"]}, "fits": fits})
    return forecasts, audit


def score_forecasts(forecasts, rows, policy):
    truth = {r["target_week_start"]: r for r in rows if r["split"] == "test"}
    keys = [(r["model"], r["target_week_start"]) for r in forecasts]
    if (len(keys) != len(set(keys)) or set(keys) != {(m, w) for m in MODELS for w in truth}
            or len(truth) != sum(r["split"] == "test" for r in rows)):
        raise ValueError("Missing, duplicate or misaligned final forecasts")
    scored = []
    for r in forecasts:
        target = truth[r["target_week_start"]]
        if (not b.finite(r["prediction"]) or r["alpha"] != policy["selected_alphas"].get(r["model"])
                or any(r[k] != target[k] for k in ("feature_week_start", "cutoff_exclusive_utc", "target_week_end"))):
            raise ValueError("Changed alpha, non-finite prediction or wrong forecast dates")
        error = r["prediction"] - target["target_ovx_mean"]
        scored.append({**r, "actual": target["target_ovx_mean"], "error": error,
                       "absolute_error": abs(error), "squared_error": error ** 2})
    return sorted(scored, key=lambda r: (r["target_week_start"], r["model"]))


def summarize_scores(scored, policy):
    results = {m: b.metrics([r for r in scored if r["model"] == m]) for m in MODELS}
    comparisons = {}
    for augmented, reference in [policy["metrics"]["primary_comparison"], *policy["metrics"]["other_comparisons"]]:
        a, ref = results[augmented], results[reference]
        comparisons[f"{augmented}_minus_{reference}"] = {
            "mae_difference": a["mae"] - ref["mae"] if a["weeks"] else None,
            "rmse_difference": a["rmse"] - ref["rmse"] if a["weeks"] else None,
            "relative_mae_reduction_percent": 100 * (ref["mae"] - a["mae"]) / ref["mae"] if ref["mae"] else None}
    return {"models": results, "comparisons": comparisons}


def paired_errors(scored):
    by_week = {}
    for r in scored:
        if r["model"] not in ("M2", "M3"):
            continue
        models = by_week.setdefault(r["target_week_start"], {})
        if r["model"] in models:
            raise ValueError("Duplicate paired error")
        models[r["model"]] = r
    result, cumulative = [], 0.
    for week, models in sorted(by_week.items()):
        if set(models) != {"M2", "M3"} or models["M2"]["actual"] != models["M3"]["actual"]:
            raise ValueError("Unpaired errors or inconsistent actuals")
        m2, m3 = models["M2"], models["M3"]
        delta = m3["absolute_error"] - m2["absolute_error"]
        cumulative += delta
        result.append({"target_week_start": week, "absolute_error_M3_minus_M2": delta,
                       "squared_error_M2": m2["squared_error"], "squared_error_M3": m3["squared_error"],
                       "cumulative_absolute_error_M3_minus_M2": cumulative})
    return result


def block_indices(weeks, block_length, replicates, seed):
    if type(block_length) is not int or block_length < 1 or type(replicates) is not int or replicates < 1:
        raise ValueError("Invalid bootstrap dimensions")
    days = [date.fromisoformat(w) for w in weeks]
    if any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError("Bootstrap weeks must be unique and ordered")
    starts = [i for i in range(len(days) - block_length + 1)
              if all(days[i + j] == days[i] + j * WEEK for j in range(block_length))]
    if not starts:
        raise ValueError("No complete calendar blocks for bootstrap")
    rng = np.random.default_rng(seed)
    selected = rng.choice(starts, size=(replicates, math.ceil(len(days) / block_length)))
    return (selected[:, :, None] + np.arange(block_length)).reshape(replicates, -1)[:, :len(days)]


def bootstrap_interval(paired, block_length, setting):
    indices = block_indices([r["target_week_start"] for r in paired], block_length,
                            setting["replicates"], setting["seed"])
    delta = np.array([r["absolute_error_M3_minus_M2"] for r in paired])
    sq2 = np.array([r["squared_error_M2"] for r in paired])
    sq3 = np.array([r["squared_error_M3"] for r in paired])
    if not all(np.isfinite(a).all() for a in (delta, sq2, sq3)) or np.any(sq2 < 0) or np.any(sq3 < 0):
        raise ValueError("Invalid bootstrap losses")
    draws_mae = delta[indices].mean(axis=1)
    draws_rmse = np.sqrt(sq3[indices].mean(axis=1)) - np.sqrt(sq2[indices].mean(axis=1))
    tail = (1 - setting["confidence"]) / 2
    def interval(draws, point):
        lower, upper = np.quantile(draws, [tail, 1 - tail], method="linear")
        return {"point": float(point), "lower": float(lower), "upper": float(upper)}
    return {"weeks": len(paired), "block_weeks": block_length, "replicates": setting["replicates"],
            "seed": setting["seed"], "confidence": setting["confidence"], "method": setting["method"],
            "mae_difference": interval(draws_mae, delta.mean()),
            "rmse_difference": interval(draws_rmse, np.sqrt(sq3.mean()) - np.sqrt(sq2.mean())),
            "draws_sha256": b.text_digest(json.dumps({"mae": draws_mae.tolist(), "rmse": draws_rmse.tolist()},
                                                     sort_keys=True, separators=(",", ":")))}


def load_inputs(root, policy):
    validate_policy(root, policy)
    dependencies = {}
    for prefix in ("dataset", "baseline", "news"):
        folder = root / policy[f"{prefix}_directory"]
        path = folder / "manifest.json"
        if b.digest(path) != policy[f"{prefix}_manifest_sha256"]:
            raise ValueError(f"Frozen {prefix} manifest changed")
        manifest = json.loads(path.read_text())
        b.verify_hashes(root, manifest["input_sha256"])
        b.verify_hashes(folder, manifest["artifact_sha256"])
        dependencies.update(manifest["input_sha256"])
        dependencies.update({str((folder / name).relative_to(root)): h for name, h in manifest["artifact_sha256"].items()})
        dependencies[str(path.relative_to(root))] = b.digest(path)
    base_dir, news_dir = root / policy["baseline_directory"], root / policy["news_directory"]
    base_selection = json.loads((base_dir / "selection.json").read_text())
    news_selection = json.loads((news_dir / "selection.json").read_text())
    selected = {"M1": base_selection["alpha"], **{m: r["alpha"] for m, r in news_selection["models"].items()}}
    if selected != policy["selected_alphas"]:
        raise ValueError("Final evaluation penalties differ from validation selections")
    weekly_policy = json.loads((root / policy["dataset_directory"] / "protocol.json").read_text())
    if weekly_policy["splits"] != policy["splits"]:
        raise ValueError("Final evaluation split differs from weekly dataset")
    for panel in policy["panels"].values():
        if (panel["ovx_lag_weeks"] != weekly_policy["timing"][panel["timing"]]["ovx_lag_weeks"]
                or panel["target_availability_delay_days"] != 7 * panel["ovx_lag_weeks"]):
            raise ValueError("Inconsistent frozen timing scenario")
    dataset = root / policy["dataset_directory"]
    # Final targets are deliberately opened here only after the protocol lock is verified.
    financial = b.read(dataset / "financial_rows.jsonl")
    panels = {name: [r for r in financial if r["eligible"]
                    and (r["timing"], r["view"]) == (setting["timing"], setting["view"])]
              for name, setting in policy["panels"].items()}
    for name, rows in panels.items():
        validate_panel(rows, policy["panels"][name], policy)
    for split in policy["splits"]:
        if [r for r in panels[policy["primary_panel"]] if r["split"] == split] != b.read(dataset / f"primary_{split}.jsonl"):
            raise ValueError("Primary split export differs from full financial panel")
    base_snapshot = json.loads((base_dir / "final_model.json").read_text())
    base_snapshot.pop("as_of_cutoff_utc")
    snapshots = {"M1": base_snapshot, **json.loads((news_dir / "final_models.json").read_text())["models"]}
    for path in (root / POLICY, root / LOCK, root / "scripts/evaluate_finance.py"):
        dependencies[str(path.relative_to(root))] = b.digest(path)
    return panels, snapshots, dependencies


def common_test_weeks(panels, primary_name):
    targets = {name: {r["target_week_start"]: (r["target_week_end"], r["target_ovx_mean"], r["target_observations"])
                     for r in rows if r["split"] == "test"} for name, rows in panels.items()}
    primary = targets[primary_name]
    for other in targets.values():
        if any(other[w] != primary[w] for w in other.keys() & primary.keys()):
            raise ValueError("Panel target values or dates disagree")
    common = sorted(set.intersection(*(set(t) for t in targets.values())))
    if not common:
        raise ValueError("No common final forecast weeks")
    return common, {name: sorted(set(t) - set(common)) for name, t in targets.items()}


def outside_crisis(row, policy):
    window = policy["crisis_exclusion"]
    return not (row["target_week_start"] <= window["end"] and row["target_week_end"] >= window["start"])


def compute(stage, root, policy):
    panels, snapshots, dependencies = load_inputs(root, policy)
    common, omitted = common_test_weeks(panels, policy["primary_panel"])
    all_scored, all_audits, summaries = [], {}, {}
    for name, rows in panels.items():
        forecasts, audit = walk_forward(rows, policy["panels"][name], policy)
        if name == policy["primary_panel"]:
            for fit in audit[0]["fits"]:
                if fit != snapshots[fit["model"]]:
                    raise ValueError("First test fit differs from frozen end-of-validation snapshot")
        scored = score_forecasts(forecasts, rows, policy)
        common_scores = [r for r in scored if r["target_week_start"] in common]
        summaries[name] = {"eligible_test_weeks": len(scored) // 4,
                           "common_test_weeks": len(common), **summarize_scores(common_scores, policy)}
        all_scored.extend({"panel": name, "in_common_evaluation": r["target_week_start"] in common, **r} for r in scored)
        all_audits[name] = audit
    primary_all = [r for r in all_scored if r["panel"] == policy["primary_panel"]]
    paired = paired_errors(primary_all)
    boot = policy["bootstrap"]
    intervals = {str(length): bootstrap_interval(paired, length, boot)
                 for length in [boot["primary_block_weeks"], *boot["sensitivity_block_weeks"]]}
    years = sorted({r["target_week_start"][:4] for r in primary_all})
    quarters = sorted({f"{r['target_week_start'][:4]}-Q{(date.fromisoformat(r['target_week_start']).month - 1) // 3 + 1}"
                       for r in primary_all})
    quarter = lambda r: f"{r['target_week_start'][:4]}-Q{(date.fromisoformat(r['target_week_start']).month - 1) // 3 + 1}"
    kept = [r for r in primary_all if outside_crisis(r, policy)]
    summary = {"version": policy["version"], "stage": "2024_2025_final_test", "final_test_scored": True,
        "retuned_from_final_test": False, "selected_alphas": policy["selected_alphas"],
        "primary_panel": policy["primary_panel"], "primary": summarize_scores(primary_all, policy),
        "bootstrap_primary_block_weeks": boot["primary_block_weeks"], "bootstrap": intervals,
        "panels_common_weeks": summaries, "common_target_weeks": common, "omitted_from_common": omitted,
        "years": {year: summarize_scores([r for r in primary_all if r["target_week_start"].startswith(year)], policy) for year in years},
        "quarters": {q: summarize_scores([r for r in primary_all if quarter(r) == q], policy) for q in quarters},
        "crisis_exclusion": {"policy": policy["crisis_exclusion"],
            "excluded_target_weeks": sorted({r["target_week_start"] for r in primary_all if not outside_crisis(r, policy)}),
            **summarize_scores(kept, policy)}, "limitations": policy["limitations"]}
    b.write_rows(stage / "test_predictions.jsonl", all_scored)
    b.save_json(stage / "fit_audit.json", all_audits)
    b.write_rows(stage / "paired_errors.jsonl", paired)
    b.write_rows(stage / "largest_errors.jsonl", sorted(primary_all, key=lambda r: (-r["absolute_error"], r["model"], r["target_week_start"])))
    b.save_json(stage / "summary.json", summary)
    b.save_json(stage / "protocol.json", policy)
    b.save_json(stage / "protocol_lock.json", json.loads((root / LOCK).read_text()))
    (stage / "report.md").write_text(report(summary), encoding="utf-8")
    return summary, dependencies


def report(summary):
    lines = ["# Final financial evaluation", "", "2024–2025 held-out forecasts with frozen settings; this test is now exposed.", "",
             "| Model | Weeks | MAE (OVX points) | RMSE (OVX points) |", "|---|---:|---:|---:|"]
    for model, result in summary["primary"]["models"].items():
        lines.append(f"| {model} | {result['weeks']} | {result['mae']:.4f} | {result['rmse']:.4f} |")
    interval = summary["bootstrap"][str(summary["bootstrap_primary_block_weeks"])]["mae_difference"]
    lines += ["", f"M3 minus M2 MAE: {interval['point']:+.4f} OVX points; paired 95% block-bootstrap interval "
              f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}]. Negative favours M3.", "",
              "| Source/timing panel | Common weeks | M0 MAE | M1 MAE | M2 MAE | M3 MAE |", "|---|---:|---:|---:|---:|---:|"]
    for panel, result in summary["panels_common_weeks"].items():
        values = " | ".join(f"{result['models'][m]['mae']:.4f}" for m in MODELS)
        lines.append(f"| {panel} | {result['common_test_weeks']} | {values} |")
    crisis = summary["crisis_exclusion"]
    lines += ["", f"Outside the predefined crisis window: {crisis['models']['M3']['weeks']} weeks; "
              f"M3 minus M2 MAE {crisis['comparisons']['M3_minus_M2']['mae_difference']:+.4f} OVX points.", "",
              *summary["limitations"], ""]
    return "\n".join(lines)


def run(root=b.ROOT, validate_only=False):
    policy = json.loads((root / POLICY).read_text())
    validate_policy(root, policy)
    output = root / policy["output_directory"]
    if not output.resolve().is_relative_to((root / "data/experiments").resolve()):
        raise ValueError("Unsafe output directory")
    if output.exists() and not validate_only:
        raise ValueError("Final evaluation exists; use --validate for exact reproduction")
    saved = None
    if validate_only:
        saved = json.loads((output / "manifest.json").read_text())
        b.verify_hashes(root, saved["input_sha256"])
        b.verify_hashes(output, saved["artifact_sha256"])
    runtime = {"python": platform.python_version(), "machine": platform.machine(),
               "packages": {k: version(k) for k in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}}
    start = {"policy_sha256": b.digest(root / POLICY), "lock_sha256": b.digest(root / LOCK),
             "implementation_sha256": {str(p.relative_to(root)): b.digest(p) for p in (
                 root / "scripts/evaluate_finance.py", root / "scripts/financial_baselines.py", root / "scripts/financial_news.py")},
             "runtime": runtime}
    output.parent.mkdir(parents=True, exist_ok=True)
    exposure_path = output.with_name(output.name + ".exposure.json")
    if exposure_path.exists():
        exposure = json.loads(exposure_path.read_text())
        if any(exposure[k] != v for k, v in start.items()):
            raise ValueError("Policy or implementation changed after final evaluation started")
    elif validate_only:
        raise ValueError("Missing final-test exposure record")
    else:
        exposure = {**start, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "final_test_evaluation_started",
                    "note": "Written before loading final financial rows; interrupted runs may resume only with identical settings and implementation."}
        with exposure_path.open("x", encoding="utf-8") as stream:
            json.dump(exposure, stream, indent=2); stream.write("\n")
    with tempfile.TemporaryDirectory(prefix=".financial_evaluation_", dir=output.parent) as temporary:
        stage = Path(temporary)
        summary, inputs = compute(stage, root, policy)
        b.save_json(stage / "exposure.json", exposure)
        inputs[str(exposure_path.relative_to(root))] = b.digest(exposure_path)
        outputs = {p.name: b.digest(p) for p in sorted(stage.iterdir()) if p.is_file()}
        b.verify_hashes(root, inputs)
        if b.digest(root / POLICY) != start["policy_sha256"]:
            raise ValueError("Final policy changed during computation")
        if validate_only:
            if (saved["input_sha256"] != inputs or saved["artifact_sha256"] != outputs
                    or saved["runtime"] != runtime or saved["schema_version"] != 1):
                raise ValueError("Final evaluation does not reproduce exactly")
        else:
            b.save_json(stage / "manifest.json", {"schema_version": 1, "status": "completed_final_evaluation",
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
