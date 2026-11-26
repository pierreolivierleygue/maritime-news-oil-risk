"""Score the frozen NLP methods once on 2023; never fit or select a model here."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

try:
    from .audit_evidence_retention import model_input
    from .compare_nlp import TARGETS, identity, metric, predict_frozen
    from .nlp_features import encode_articles
    from .prepare_modelling_dataset import text_digest
    from .prepare_news_review import ROOT, digest, read, write_rows
    from .reserve_news_evaluation import save_json, validate as validate_reservation
    from .validate_evaluation_annotations import IDENTITY, validate as validate_references, verify_hashes
except ImportError:
    from audit_evidence_retention import model_input
    from compare_nlp import TARGETS, identity, metric, predict_frozen
    from nlp_features import encode_articles
    from prepare_modelling_dataset import text_digest
    from prepare_news_review import ROOT, digest, read, write_rows
    from reserve_news_evaluation import save_json, validate as validate_reservation
    from validate_evaluation_annotations import IDENTITY, validate as validate_references, verify_hashes

POLICY = "config/nlp_evaluation_v1.json"


def load_frozen(root, policy):
    verify_hashes(root, policy["frozen_manifest_sha256"])
    development = root / "data/experiments" / policy["development_experiment"]
    manifest = json.loads((development / "manifest.json").read_text())
    verify_hashes(root, manifest["input_sha256"])
    verify_hashes(development, manifest["artifact_sha256"])
    runtime = {"packages": {k: version(k) for k in manifest["runtime"]["packages"]},
               "python": platform.python_version(), "machine": platform.machine()}
    if runtime != manifest["runtime"]:
        raise ValueError("Frozen-model runtime changed")
    config = json.loads((development / "protocol.json").read_text())
    selection = json.loads((development / "selection.json").read_text())
    models = json.loads((development / "final_models.json").read_text())
    if (policy["primary_method"] != selection["method"] or policy["targets"] != list(TARGETS)
            or policy["methods"] != config["methods"] or set(models) != set(policy["methods"])):
        raise ValueError("Frozen primary method or comparison set changed")
    primary = models[policy["primary_method"]]
    if primary["c"] != selection["c"] or primary["threshold"] != selection["threshold"]:
        raise ValueError("Primary hyperparameters differ from frozen selection")
    audit = root / "data/audit" / policy["input_audit_id"]
    input_manifest = json.loads((audit / "manifest.json").read_text())
    verify_hashes(root, input_manifest["input_sha256"])
    verify_hashes(audit, input_manifest["artifact_sha256"])
    rule = json.loads((audit / "selected_rule.json").read_text())
    if (rule["rule_version"] != selection["input_rule"] or rule["selected_candidate"] != "full"
            or rule["body_word_limit"] is not None or rule["include_title"] is not True
            or rule["separator"] != "\n\n" or rule["model_tokenizer_truncation_allowed"] is not False):
        raise ValueError("Frozen full-text input rule changed")
    return config, selection, models, runtime


def prepare_inputs(queue, texts, training):
    """Join reservation identities and source versions without accepting labels."""
    by_key = {(r["source"], r["document_id"]): r for r in texts}
    keys = [(r["source"], r["document_id"]) for r in queue]
    if not queue or len(by_key) != len(texts) or len(set(keys)) != len(queue) or set(keys) != set(by_key):
        raise ValueError("Evaluation queue/text identity mismatch")
    train_keys = {(r["source"], r["document_id"]) for r in training}
    train_groups = {r["split_group_id"] for r in training}
    seen_families, inputs = set(), []
    for number, q in enumerate(queue, 1):
        d = by_key[q["source"], q["document_id"]]
        if (any(q[k] != d[k] for k in IDENTITY) or q["article_number"] != number
                or q["status"] != "reserved_unlabelled"
                or any(k in q for k in (*TARGETS, "deescalation", "labels"))):
            raise ValueError("Evaluation version/order or unlabelled-queue mismatch")
        if (q["nlp_training_eligible"] is not False or q["partition"] != "evaluation_reserved"
                or q["evaluation_selected"] is not True or q["evaluation_eligible"] is not True
                or not "2023-01-01" <= q["publication_date"] < "2024-01-01"):
            raise ValueError("Invalid holdout reservation eligibility/date")
        if ((q["source"], q["document_id"]) in train_keys or q["split_group_id"] in train_groups):
            raise ValueError("Evaluation overlaps frozen training identities/components")
        if q["near_family_id"] in seen_families:
            raise ValueError("Repeated evaluation family")
        seen_families.add(q["near_family_id"])
        if text_digest(d["body_clean"]) != q["body_hash"]:
            raise ValueError("Evaluation body hash mismatch")
        inputs.append({**{k: q[k] for k in (*IDENTITY, "split_group_id", "article_number")},
                       "input_rule": "nlp_text_input_1", "body_model_sha256": q["body_hash"],
                       **model_input(d["title"], d["body_clean"], None)})
    return inputs


def predict_methods(inputs, matrix, models, config, methods):
    texts = [r["input_text"] for r in inputs]
    results = []
    for method in methods:
        probs, predictions = predict_frozen(models[method], texts, matrix, config)
        if probs.shape != (len(inputs), len(TARGETS)) or predictions.shape != probs.shape:
            raise ValueError("Prediction shape mismatch")
        for i, row in enumerate(inputs):
            results.append({"method": method, **{k: row[k] for k in (
                *IDENTITY, "split_group_id", "article_number", "input_sha256")},
                "probabilities": {t: float(probs[i, j]) for j, t in enumerate(TARGETS)},
                "predictions": {t: int(predictions[i, j]) for j, t in enumerate(TARGETS)},
                "target_modes": {t: models[method]["targets"][t]["mode"] for t in TARGETS}})
    return results


def join_references(predictions, references, methods):
    by_key = {(r["source"], r["document_id"]): r for r in references}
    if len(by_key) != len(references) or not references:
        raise ValueError("Missing or repeated references")
    for method in methods:
        selected = [p for p in predictions if p["method"] == method]
        keys = [(p["source"], p["document_id"]) for p in selected]
        if len(keys) != len(set(keys)) or set(keys) != set(by_key):
            raise ValueError("Missing or extra predictions for a method")
    if set(p["method"] for p in predictions) != set(methods):
        raise ValueError("Unexpected evaluated method")
    scored = []
    for p in predictions:
        a = by_key[p["source"], p["document_id"]]
        if any(p[k] != a[k] for k in (*IDENTITY, "split_group_id", "article_number")):
            raise ValueError("Prediction/reference version or component mismatch")
        labels = {t: a[t] for t in TARGETS}
        if any(v is not None and (type(v) is not int or v not in (0, 1)) for v in labels.values()):
            raise ValueError("Invalid reference target")
        scored.append({**p, "labels": labels})
    return scored


def score_rows(rows):
    targets = {t: {**metric([r["labels"][t] for r in rows], [r["predictions"][t] for r in rows]),
                   "unresolved": sum(r["labels"][t] is None for r in rows)} for t in TARGETS}
    scores = [m["f1"] for m in targets.values() if m["f1"] is not None]
    return {"articles": len(rows), "families": len({r["near_family_id"] for r in rows}),
            "related_components": len({r["split_group_id"] for r in rows}),
            "targets": targets, "macro_positive_f1": float(np.mean(scores)) if scores else None}


def summarize(scored, queue, policy):
    counts = Counter(r["split_group_id"] for r in queue)
    dominant = min(counts, key=lambda group: (-counts[group], group))
    methods = {}
    for method in policy["methods"]:
        rows = [r for r in scored if r["method"] == method]
        quarter = lambda r: (int(r["publication_date"][5:7]) - 1) // 3 + 1
        methods[method] = {"role": "primary" if method == policy["primary_method"] else "frozen_alternative",
            "overall": score_rows(rows),
            "without_dominant_component": score_rows([r for r in rows if r["split_group_id"] != dominant]),
            "dominant_component_only": score_rows([r for r in rows if r["split_group_id"] == dominant]),
            "by_component": {g: score_rows([r for r in rows if r["split_group_id"] == g]) for g in sorted(counts)},
            "by_source": {s: score_rows([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})},
            "by_quarter": {str(q): score_rows([r for r in rows if quarter(r) == q]) for q in range(1, 5)}}
    return {"protocol_version": policy["protocol_version"], "primary_method": policy["primary_method"],
            "methods": methods, "dominant_component": {"id": dominant, "articles": counts[dominant]},
            "selection_changed": False, "models_refitted": False, "financial_outcomes_loaded": False,
            "interpretation": "Held-out agreement with AI references; not independently verified event accuracy.",
            "uncertainty": policy["uncertainty"]}


def report_markdown(summary):
    lines = ["# Frozen 2023 NLP evaluation", "",
             f"Primary method remains **{summary['primary_method']}**, selected before held-out scoring.", "",
             "| Method | Role | Relevance F1 | Threat F1 | Disruption F1 | Mean F1 |",
             "|---|---|---:|---:|---:|---:|"]
    for name, result in summary["methods"].items():
        overall = result["overall"]
        f1s = " | ".join(f"{overall['targets'][t]['f1']:.3f}" for t in TARGETS)
        lines.append(f"| {name} | {result['role']} | {f1s} | {overall['macro_positive_f1']:.3f} |")
    lines += ["", "## Excluding the dominant related-report component", "",
              "| Method | Articles | Relevance F1 | Threat F1 | Disruption F1 | Mean F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, result in summary["methods"].items():
        outside = result["without_dominant_component"]
        f1s = " | ".join(f"{outside['targets'][t]['f1']:.3f}" for t in TARGETS)
        lines.append(f"| {name} | {outside['articles']} | {f1s} | {outside['macro_positive_f1']:.3f} |")
    lines += ["", summary["interpretation"], "", summary["uncertainty"], "",
              "Per-target precision, recall, confusion counts and source/quarter/component results are in summary.json.",
              "No settings, training labels or primary-method selection changed after opening these scores.", ""]
    return "\n".join(lines)


def compute(stage, root, policy):
    config, selection, models, runtime = load_frozen(root, policy)
    validate_reservation(root, policy["reservation_id"])
    reservation = root / "data/evaluation" / policy["reservation_id"]
    queue = read(reservation / "queue_snapshot.jsonl")
    texts = read(reservation / "evaluation_texts.jsonl")
    counts = Counter(r["split_group_id"] for r in queue)
    if (len(queue) != policy["expected_articles"] or len(counts) != policy["expected_components"]
            or max(counts.values()) != policy["expected_dominant_component_articles"]):
        raise ValueError("Reserved evaluation population changed")
    inputs = prepare_inputs(queue, texts, selection["training_inputs"])
    matrix, coverage = encode_articles(inputs, config["embedding"], root)
    predictions = predict_methods(inputs, matrix, models, config, policy["methods"])
    write_rows(stage / "inputs.jsonl", inputs)
    write_rows(stage / "predictions.jsonl", predictions)
    np.save(stage / "embeddings.npy", matrix, allow_pickle=False)
    save_json(stage / "embedding_coverage.json", coverage)
    commit = {"primary_method": policy["primary_method"], "reference_join_performed": False,
              "training_performed": False, "input_sha256": digest(stage / "inputs.jsonl"),
              "predictions_sha256": digest(stage / "predictions.jsonl"),
              "frozen_manifest_sha256": policy["frozen_manifest_sha256"]}
    save_json(stage / "prediction_commit.json", commit)

    # Predictions and their content hashes are on disk before references are parsed.
    validate_references(root, policy["reservation_id"])
    references = read(reservation / "labelling_v1/annotations.jsonl")
    scored = join_references(predictions, references, policy["methods"])
    if digest(stage / "predictions.jsonl") != commit["predictions_sha256"]:
        raise ValueError("Predictions changed after reference join")
    summary = summarize(scored, queue, policy)
    summary["embeddings"] = {"shape": list(matrix.shape), "chunks": sum(len(c["chunks"]) for c in coverage),
                             "content_tokens": sum(c["tokens"] for c in coverage), "all_tokens_covered_once": True}
    errors = [{**{k: r[k] for k in ("method", "source", "document_id", "record_version_hash", "title",
                                    "publication_date", "split_group_id")},
               "target": t, "reference": r["labels"][t], "prediction": r["predictions"][t],
               "probability": r["probabilities"][t], "mode": r["target_modes"][t]}
              for r in scored for t in TARGETS
              if r["labels"][t] is not None and r["labels"][t] != r["predictions"][t]]
    write_rows(stage / "scored_predictions.jsonl", scored)
    write_rows(stage / "errors.jsonl", errors)
    save_json(stage / "protocol.json", policy)
    save_json(stage / "summary.json", summary)
    save_json(stage / "selection_preserved.json", {"primary_method": policy["primary_method"],
              "source_selection_sha256": digest(root / "data/experiments" / policy["development_experiment"] / "selection.json"),
              "evaluation_completed": True, "development_selection_changed": False, "heldout_refit": False})
    (stage / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    paths = [root / POLICY, root / "scripts/evaluate_nlp_holdout.py", root / "scripts/validate_evaluation_annotations.py",
             reservation / "queue_snapshot.jsonl", reservation / "evaluation_texts.jsonl",
             reservation / "labelling_v1/annotations.jsonl"]
    paths += [root / name for name in policy["frozen_manifest_sha256"]]
    return summary, {str(p.relative_to(root)): digest(p) for p in paths}, runtime


def run(root=ROOT, validate_only=False):
    policy = json.loads((root / POLICY).read_text())
    output = root / policy["output_directory"]
    if not output.resolve().is_relative_to((root / "data/experiments").resolve()):
        raise ValueError("Unsafe output directory")
    if output.exists() and not validate_only:
        raise ValueError("Evaluation already recorded; only --validate may reproduce the frozen run")
    saved = None
    if validate_only:
        saved = json.loads((output / "manifest.json").read_text())
        # Check code, policy, references and outputs before re-exposing metrics.
        verify_hashes(root, saved["input_sha256"])
        verify_hashes(output, saved["artifact_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nlp_holdout_", dir=output.parent) as temporary:
        stage = Path(temporary)
        initial_policy_hash = digest(root / POLICY)
        summary, inputs, runtime = compute(stage, root, policy)
        outputs = {p.name: digest(p) for p in sorted(stage.iterdir()) if p.is_file()}
        verify_hashes(root, inputs)
        verify_hashes(root, policy["frozen_manifest_sha256"])
        if digest(root / POLICY) != initial_policy_hash:
            raise ValueError("Evaluation policy changed during run")
        if validate_only:
            if (saved["runtime"] != runtime or saved["input_sha256"] != inputs
                    or saved["artifact_sha256"] != outputs or saved["schema_version"] != 1):
                raise ValueError("Frozen holdout computation does not reproduce exactly")
        else:
            save_json(stage / "manifest.json", {"schema_version": 1,
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "status": "completed_frozen_evaluation", "runtime": runtime,
                      "input_sha256": inputs, "artifact_sha256": outputs})
            output.mkdir(exist_ok=False)
            for path in sorted(stage.iterdir()):
                path.rename(output / path.name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(report_markdown(run(validate_only=args.validate)))
