"""Compare three text methods on purged chronological development folds only.

No held-out reference labels are read. All reported scores are development
selection diagnostics, including explicitly identified small-class rule fallbacks.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import tempfile
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

try:
    from .audit_evidence_retention import run as validate_inputs
    from .nlp_features import embedding_cache, keyword_predictions
    from .prepare_news_review import ROOT, digest, read, write_rows
    from .reserve_news_evaluation import save_json
except ImportError:
    from audit_evidence_retention import run as validate_inputs
    from nlp_features import embedding_cache, keyword_predictions
    from prepare_news_review import ROOT, digest, read, write_rows
    from reserve_news_evaluation import save_json

POLICY = "config/nlp_comparison_v1.json"
TARGETS = ("relevant", "threat", "disruption")


def identity(row):
    return {k: row[k] for k in ("source", "document_id", "record_version_hash", "split_group_id")}


def make_folds(rows, years):
    if len({(r["source"], r["document_id"]) for r in rows}) != len(rows):
        raise ValueError("Repeated development document")
    if any(not "2019-01-01" <= r["publication_date"] < "2023-01-01"
           or r["modelling_role"] != "primary" or not r["primary_training_eligible"] for r in rows):
        raise ValueError("Ineligible development row")
    folds = []
    for year in years:
        validation = [i for i, r in enumerate(rows) if r["publication_date"][:4] == str(year)]
        groups = {rows[i]["split_group_id"] for i in validation}
        earlier = [i for i, r in enumerate(rows) if r["publication_date"] < f"{year}-01-01"]
        train = [i for i in earlier if rows[i]["split_group_id"] not in groups]
        purged = [i for i in earlier if i not in train]
        if not train or not validation:
            raise ValueError("Empty purged chronological fold")
        if (max(rows[i]["publication_date"] for i in train) >= min(rows[i]["publication_date"] for i in validation)
                or {rows[i]["split_group_id"] for i in train} & groups):
            raise ValueError("Temporal or related-report leakage")
        folds.append({"year": year, "train_indices": train, "validation_indices": validation,
                      "purged_indices": purged,
                      "train": [identity(rows[i]) for i in train],
                      "validation": [identity(rows[i]) for i in validation],
                      "purged": [identity(rows[i]) for i in purged],
                      "class_counts": {part: {t: dict(Counter(
                          "null" if rows[i]["labels"][t] is None else str(rows[i]["labels"][t])
                          for i in ids)) for t in TARGETS}
                          for part, ids in (("train", train), ("validation", validation))}})
    return folds


def tfidf_parameters(config):
    return {**config["tfidf"], "ngram_range": tuple(config["tfidf"]["ngram_range"])}


def features(method, texts, embeddings, train, validation, config):
    if method == "embedding":
        return embeddings[train], embeddings[validation], None
    vectorizer = TfidfVectorizer(**tfidf_parameters(config))
    training = vectorizer.fit_transform([texts[i] for i in train])
    return training, vectorizer.transform([texts[i] for i in validation]), vectorizer


def fit_targets(training, rows, train, c, config):
    models = {}
    for target in TARGETS:
        resolved = [j for j, i in enumerate(train) if rows[i]["labels"][target] is not None]
        labels = np.array([rows[train[j]]["labels"][target] for j in resolved])
        counts = Counter(labels.tolist())
        if min(counts.get(0, 0), counts.get(1, 0)) < config["minimum_class_count"]:
            models[target] = {"mode": "keyword_fallback", "class_counts": dict(counts),
                              "reason": "insufficient_resolved_training_class_count"}
            continue
        model = LogisticRegression(C=c, random_state=config["seed"], **config["logistic"])
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(training[resolved], labels)
        models[target] = {"mode": "logistic", "class_counts": dict(counts),
                          "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                          "n_iter": int(model.n_iter_[0])}
    return models


def probabilities(matrix, models, rules):
    result = np.zeros((len(rules), len(TARGETS)))
    for j, target in enumerate(TARGETS):
        model = models[target]
        if model["mode"] == "logistic":
            result[:, j] = expit(matrix @ np.array(model["coef"]) + model["intercept"])
        else:
            result[:, j] = [r[target] for r in rules]
    return result


def constrain_predictions(probs, threshold):
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Invalid prediction probabilities")
    result = (probs >= threshold).astype(int)
    result[:, 1:] *= result[:, :1]
    return result


def metric(y, predicted):
    pairs = [(a, b) for a, b in zip(y, predicted) if a is not None]
    tp = sum(a == b == 1 for a, b in pairs)
    fp = sum(a == 0 and b == 1 for a, b in pairs)
    fn = sum(a == 1 and b == 0 for a, b in pairs)
    tn = sum(a == b == 0 for a, b in pairs)
    if not pairs:
        return {"n": 0, "positives": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "precision": None, "recall": None, "f1": None}
    return {"n": len(pairs), "positives": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def score(predictions):
    targets = {t: metric([r["labels"][t] for r in predictions], [r["predictions"][t] for r in predictions])
               for t in TARGETS}
    return {"targets": targets, "macro_positive_f1":
            float(np.mean([v["f1"] for v in targets.values() if v["f1"] is not None])) if predictions else None}


def summarize(predictions, dominant):
    return {"overall": score(predictions),
            "by_year": {str(y): score([r for r in predictions if r["year"] == y])
                        for y in sorted({r["year"] for r in predictions})},
            "by_source": {s: score([r for r in predictions if r["source"] == s])
                          for s in sorted({r["source"] for r in predictions})},
            "without_largest_development_component": score([r for r in predictions if r["split_group_id"] != dominant]),
            "by_target_prediction_mode": {t: {mode: metric(
                [r["labels"][t] for r in predictions if r["target_modes"][t] == mode],
                [r["predictions"][t] for r in predictions if r["target_modes"][t] == mode])
                for mode in ("keyword", "keyword_fallback", "logistic")} for t in TARGETS}}


def candidate_key(candidate):
    return (-candidate["metrics"]["overall"]["macro_positive_f1"],
            ("keyword", "tfidf", "embedding").index(candidate["method"]),
            candidate["c"] or 0, abs((candidate["threshold"] or .5) - .5), candidate["threshold"] or .5)


def prediction_rows(rows, indices, year, probs, models, threshold):
    predictions = constrain_predictions(probs, threshold)
    return [{**identity(rows[i]), "year": year, "labels": rows[i]["labels"],
             "probabilities": {t: float(probs[j, k]) for k, t in enumerate(TARGETS)},
             "predictions": {t: int(predictions[j, k]) for k, t in enumerate(TARGETS)},
             "target_modes": {t: models[t]["mode"] for t in TARGETS}}
            for j, i in enumerate(indices)]


def safe_transform(saved, texts, config):
    vectorizer = TfidfVectorizer(**tfidf_parameters(config), vocabulary=saved["vocabulary"])
    vectorizer.idf_ = np.array(saved["idf"])
    return vectorizer.transform(texts)


def predict_frozen(bundle, texts, embeddings, config):
    """Safe JSON model parameters; no pickle and no refitting at prediction time."""
    rules = [keyword_predictions(t, config["keyword"]) for t in texts]
    if bundle["method"] == "keyword":
        probs = np.array([[r[t] for t in TARGETS] for r in rules], dtype=float)
    else:
        matrix = embeddings if bundle["method"] == "embedding" else safe_transform(bundle["vectorizer"], texts, config)
        probs = probabilities(matrix, bundle["targets"], rules)
    return probs, constrain_predictions(probs, bundle["threshold"] or .5)


def compute(root=ROOT, recompute_embeddings=False):
    config = json.loads((root / POLICY).read_text())
    if config["targets"] != list(TARGETS) or config["validation_years"] != [2020, 2021, 2022]:
        raise ValueError("Unexpected target or date protocol")
    validate_inputs(root, audit_id=config["input_audit_id"], validate_only=True)
    dataset = root / "data/processed" / config["dataset_id"]
    audit = root / "data/audit" / config["input_audit_id"]
    rows = read(dataset / "development_primary.jsonl")
    inputs = read(audit / "inputs_primary.jsonl")
    if len(rows) != len(inputs) or any(identity(r) != identity(i) for r, i in zip(rows, inputs)):
        raise ValueError("Model input/reference join mismatch")
    texts = [r["input_text"] for r in inputs]
    folds = make_folds(rows, config["validation_years"])
    embeddings, coverage, cache = embedding_cache(inputs, config["embedding"], root, recompute_embeddings)
    rules = [keyword_predictions(text, config["keyword"]) for text in texts]
    group_counts = Counter(r["split_group_id"] for r in rows)
    dominant = min(group_counts, key=lambda g: (-group_counts[g], g))
    candidates, prediction_sets, fold_models = [], {}, []
    for method in config["methods"]:
        method_cs = [None] if method == "keyword" else config["c_grid"]
        method_thresholds = [None] if method == "keyword" else config["threshold_grid"]
        feature_sets = {} if method == "keyword" else {f["year"]: features(
            method, texts, embeddings, f["train_indices"], f["validation_indices"], config) for f in folds}
        for c in method_cs:
            fold_predictions = []
            for fold in folds:
                tr, va = fold["train_indices"], fold["validation_indices"]
                if method == "keyword":
                    models = {t: {"mode": "keyword"} for t in TARGETS}
                    probs = np.array([[rules[i][t] for t in TARGETS] for i in va], dtype=float)
                else:
                    train_x, val_x, vectorizer = feature_sets[fold["year"]]
                    models = fit_targets(train_x, rows, tr, c, config)
                    probs = probabilities(val_x, models, [rules[i] for i in va])
                    fold_models.append({"method": method, "c": c, "year": fold["year"],
                                        "training_features": int(train_x.shape[1]),
                                        "targets": {t: {k: v for k, v in m.items() if k not in ("coef", "intercept")}
                                                    for t, m in models.items()},
                                        "vocabulary": sorted(vectorizer.vocabulary_) if vectorizer is not None else None})
                fold_predictions.append((fold, probs, models))
            for threshold in method_thresholds:
                key = f"{method}_c{c}_t{threshold}"
                predictions = [p for f, probs, models in fold_predictions for p in prediction_rows(
                    rows, f["validation_indices"], f["year"], probs, models, threshold or .5)]
                prediction_sets[key] = predictions
                candidates.append({"candidate_id": key, "method": method, "c": c,
                                   "threshold": threshold, "metrics": summarize(predictions, dominant)})
    best = {m: min([c for c in candidates if c["method"] == m], key=candidate_key) for m in config["methods"]}
    winner = min(best.values(), key=candidate_key)
    final_models = {}
    full = list(range(len(rows)))
    for method, candidate in best.items():
        bundle = {"method": method, "c": candidate["c"], "threshold": candidate["threshold"],
                  "training_articles": len(rows), "vectorizer": None}
        if method == "keyword":
            bundle["targets"] = {t: {"mode": "keyword"} for t in TARGETS}
        else:
            train_x, _, vectorizer = features(method, texts, embeddings, full, full, config)
            bundle["targets"] = fit_targets(train_x, rows, full, candidate["c"], config)
            if vectorizer is not None:
                bundle["vectorizer"] = {"vocabulary": {k: int(v) for k, v in sorted(vectorizer.vocabulary_.items())},
                                        "idf": vectorizer.idf_.tolist()}
        final_models[method] = bundle
    selected_predictions = [{"method": method, **p} for method, c in best.items()
                            for p in prediction_sets[c["candidate_id"]]]
    errors = [{"method": r["method"], **{k: r[k] for k in ("source", "document_id", "record_version_hash", "year")},
               "target": target, "reference": r["labels"][target], "prediction": r["predictions"][target],
               "probability": r["probabilities"][target], "mode": r["target_modes"][target]}
              for r in selected_predictions for target in TARGETS
              if r["labels"][target] is not None and r["labels"][target] != r["predictions"][target]]
    summary = {"protocol_version": config["protocol_version"], "primary_articles": len(rows),
               "validation_articles": sum(len(f["validation_indices"]) for f in folds),
               "best_by_method": best, "selected_candidate": winner,
               "score_interpretation": "Development model/threshold selection scores; agreement with AI references, not held-out estimates.",
               "evaluation_labels_loaded": False, "financial_outcomes_loaded": False,
               "largest_development_component": {"id": dominant, "articles": group_counts[dominant]},
               "embeddings": {"shape": list(embeddings.shape), "tokens": sum(r["tokens"] for r in coverage),
                              "chunks": sum(len(r["chunks"]) for r in coverage), "all_tokens_covered_once": True},
               "fold_counts": [{"year": f["year"], "train": len(f["train_indices"]),
                                "purged": len(f["purged_indices"]), "validation": len(f["validation_indices"]),
                                "class_counts": f["class_counts"]} for f in folds]}
    outputs = {"protocol.json": config, "folds.json": folds, "candidate_scores.json": candidates,
               "fold_fit_audit.json": fold_models, "selected_predictions.jsonl": selected_predictions,
               "development_errors.jsonl": errors, "final_models.json": final_models, "summary.json": summary,
               "selection.json": {"method": winner["method"], "c": winner["c"], "threshold": winner["threshold"],
                                  "training_inputs": [{**identity(r), "input_sha256": i["input_sha256"]} for r, i in zip(rows, inputs)],
                                  "holdout_scored": False, "primary_metric": "macro_positive_f1",
                                  "model_file": "final_models.json", "input_rule": "nlp_text_input_1"}}
    paths = [root / POLICY, root / "pyproject.toml", root / "uv.lock", root / "scripts/compare_nlp.py",
             root / "scripts/nlp_features.py", root / "scripts/download_nlp_encoder.py",
             dataset / "manifest.json", dataset / "development_primary.jsonl",
             audit / "manifest.json", audit / "inputs_primary.jsonl", audit / "selected_rule.json",
             cache / "manifest.json", cache / "embeddings.npy", cache / "coverage.json",
             root / config["embedding"]["model_directory"] / "encoder_manifest.json"]
    return outputs, {str(p.relative_to(root)): digest(p) for p in paths}


def run(root=ROOT, experiment_id="nlp_dev_comparison_v1", validate_only=False, recompute_embeddings=False):
    if not re.fullmatch(r"[a-z0-9_]+", experiment_id):
        raise ValueError("Invalid experiment ID")
    output = root / "data/experiments" / experiment_id
    if output.exists() and not validate_only:
        raise ValueError("Existing experiment left unchanged; use --validate or a new --experiment-id")
    if validate_only and not output.is_dir():
        raise ValueError("Experiment does not exist")
    artifacts, hashes = compute(root, recompute_embeddings)
    versions = {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "sentence-transformers",
                                                "transformers", "torch", "tokenizers")}
    runtime = {"packages": versions, "python": platform.python_version(), "machine": platform.machine()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nlp_compare_", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, value in artifacts.items():
            if name.endswith(".jsonl"):
                write_rows(stage / name, value)
            else:
                save_json(stage / name, value)
        outputs = {name: digest(stage / name) for name in artifacts}
        if any(digest(root / name) != expected for name, expected in hashes.items()):
            raise ValueError("Inputs changed during experiment")
        if validate_only:
            manifest = json.loads((output / "manifest.json").read_text())
            if (manifest["experiment_id"] != experiment_id or manifest["runtime"] != runtime
                    or manifest["input_sha256"] != hashes or manifest["artifact_sha256"] != outputs
                    or any(digest(output / name) != expected for name, expected in outputs.items())):
                raise ValueError("Experiment/input drift or non-reproducible fit")
        else:
            save_json(stage / "manifest.json", {"experiment_id": experiment_id, "schema_version": 1,
                      "created_at": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
                      "input_sha256": hashes, "artifact_sha256": outputs})
            output.mkdir(exist_ok=False)
            for path in sorted(stage.iterdir()):
                path.rename(output / path.name)
    return artifacts["summary.json"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="nlp_dev_comparison_v1")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--recompute-embeddings", action="store_true")
    args = parser.parse_args()
    result = run(experiment_id=args.experiment_id, validate_only=args.validate,
                 recompute_embeddings=args.recompute_embeddings)
    print(json.dumps({"validation_articles": result["validation_articles"],
          "best_by_method": {m: {"c": v["c"], "threshold": v["threshold"], "metrics": v["metrics"]["overall"]}
                             for m, v in result["best_by_method"].items()},
          "selected_method": result["selected_candidate"]["method"], "embeddings": result["embeddings"]}, indent=2))
