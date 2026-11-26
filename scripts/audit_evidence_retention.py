"""Audit development evidence retention and materialize an exact text-input rule."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import tempfile

try:
    from .prepare_modelling_dataset import run as validate_dataset, text_digest
    from .prepare_news_review import ROOT, digest, read, write_rows
    from .reserve_news_evaluation import save_json
except ImportError:
    from prepare_modelling_dataset import run as validate_dataset, text_digest
    from prepare_news_review import ROOT, digest, read, write_rows
    from reserve_news_evaluation import save_json

POLICY = "config/evidence_retention_v1.json"
TARGETS = ("relevant", "threat", "disruption", "deescalation")


def candidate_id(limit):
    return "full" if limit is None else f"head_{limit}"


def model_input(title, body, limit):
    """Apply an article-independent prefix rule; never use labels or evidence."""
    if not isinstance(title, str) or not title.strip() or not isinstance(body, str) or not body.strip():
        raise ValueError("Missing title or body")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("Word limit must be a nonnegative integer or null")
    words = list(re.finditer(r"\S+", body))
    if limit is None or limit >= len(words):
        end = len(body)
    elif limit == 0:
        end = 0
    else:
        end = words[limit - 1].end()
    prefix = body[:end]
    text = title + ("\n\n" + prefix if prefix else "")
    return {"input_text": text, "input_sha256": text_digest(text),
            "body_end_exclusive": end, "body_start_in_input": len(title) + 2 if prefix else None,
            "body_words_total": len(words), "body_words_retained": len(re.findall(r"\S+", prefix)),
            "input_words": len(re.findall(r"\S+", text)), "input_characters": len(text)}


def audit_row(row, limit):
    if (row["modelling_role"] not in ("primary", "supplementary")
            or row["nlp_training_eligible"] is not True
            or not "2019-01-01" <= row["publication_date"] < "2023-01-01"):
        raise ValueError("Ineligible record in development retention audit")
    body = row["body_model"]
    if text_digest(body) != row["body_model_sha256"]:
        raise ValueError("Derived body hash mismatch")
    inp = model_input(row["title"], body, limit)
    end = inp["body_end_exclusive"]
    spans = row["evidence_spans_model"]
    if not spans:
        raise ValueError("Missing evidence; cannot score as vacuous success")
    states = []
    for i, span in enumerate(spans):
        start, stop = span["start"], span["end"]
        if (span["field"] != "body_model" or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= len(body) or body[start:stop] != span["text"]):
            raise ValueError("Evidence offset/text mismatch")
        state = "retained" if stop <= end else ("partial" if start < end else "absent")
        states.append({"span_index": i, "start": start, "end": stop, "status": state,
                       "retained_characters": max(0, min(end, stop) - start)})
    retained = sum(s["status"] == "retained" for s in states)
    return {**{k: row[k] for k in ("source", "document_id", "record_version_hash",
                                   "split_group_id", "publication_date", "modelling_role", "labels")},
            "candidate_id": candidate_id(limit), "body_word_limit": limit,
            **{k: v for k, v in inp.items() if k != "input_text"},
            "span_count": len(states), "retained_span_count": retained,
            "all_spans_retained": retained == len(states), "any_span_retained": retained > 0,
            "spans": states}


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split_group_id"]].append(row)
    words = sorted(r["input_words"] for r in rows)
    all_count = sum(r["all_spans_retained"] for r in rows)
    any_count = sum(r["any_span_retained"] for r in rows)
    spans = sum(r["span_count"] for r in rows)
    retained = sum(r["retained_span_count"] for r in rows)
    return {"articles": len(rows), "articles_all_spans_retained": all_count,
            "articles_any_span_retained": any_count, "articles_no_complete_span": len(rows) - any_count,
            "article_all_retention": fraction(all_count, len(rows)),
            "article_any_retention": fraction(any_count, len(rows)),
            "spans": spans, "retained_spans": retained, "span_retention": fraction(retained, spans),
            "partial_spans": sum(s["status"] == "partial" for r in rows for s in r["spans"]),
            "related_components": len(groups),
            "components_with_all_articles_retaining_all_spans": sum(
                all(r["all_spans_retained"] for r in members) for members in groups.values()),
            "input_words": {"total": sum(words), "mean": statistics.mean(words) if words else None,
                            "median": statistics.median(words) if words else None,
                            "p95_nearest_rank": words[math.ceil(.95 * len(words)) - 1] if words else None,
                            "maximum": max(words) if words else None}}


def stratify(rows, dominant_group):
    return {"overall": aggregate(rows),
            "by_source": {s: aggregate([r for r in rows if r["source"] == s])
                          for s in sorted({r["source"] for r in rows})},
            "by_year": {y: aggregate([r for r in rows if r["publication_date"][:4] == y])
                        for y in sorted({r["publication_date"][:4] for r in rows})},
            "positive_label_articles": {target: aggregate([r for r in rows if r["labels"][target] == 1])
                                        for target in TARGETS},
            "negative_relevance_articles": aggregate([r for r in rows if r["labels"]["relevant"] == 0]),
            "without_largest_primary_component": aggregate(
                [r for r in rows if r["split_group_id"] != dominant_group])}


def choose_limit(primary_audits, limits):
    """Selection uses primary article evidence only, with a full-text fallback."""
    for limit in [n for n in limits if n is not None and n > 0] + [None]:
        rows = [r for r in primary_audits if r["body_word_limit"] == limit]
        if rows and all(r["all_spans_retained"] for r in rows):
            return limit
    raise ValueError("Even full body does not retain all reference evidence")


def report_markdown(summary):
    lines = ["# Development evidence-retention audit", "",
             "Exact retention of recorded body spans; not semantic sufficiency or classifier performance.", "",
             "| Body word limit | All spans: articles | Any span: articles | Complete spans | Mean input words |",
             "|---|---:|---:|---:|---:|"]
    for candidate in summary["candidates"]:
        r = candidate["primary"]["overall"]
        limit = candidate["body_word_limit"]
        lines.append(f"| {'Full body' if limit is None else limit} | "
                     f"{r['articles_all_spans_retained']}/{r['articles']} | "
                     f"{r['articles_any_span_retained']}/{r['articles']} | "
                     f"{r['retained_spans']}/{r['spans']} | {r['input_words']['mean']:.1f} |")
    lines += ["", f"Selected: **{summary['selected_candidate']}**, always with the unchanged headline.", "",
              "Primary data select the rule; supplementary results are separate in summary.json.",
              "article_audit.jsonl identifies every dropped or partial span for every candidate.", "",
              "## Interpretation", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    return "\n".join(lines) + "\n"


def assemble(root=ROOT):
    policy = json.loads((root / POLICY).read_text())
    limits = policy["candidate_body_word_limits"]
    finite = [n for n in limits if n is not None]
    if (limits.count(None) != 1 or limits[-1] is not None
            or any(type(n) is not int or n < 0 for n in finite)
            or finite != sorted(set(finite)) or not finite or finite[0] != 0
            or policy["selection_target"] != "all_recorded_spans_on_all_primary_articles"):
        raise ValueError("Invalid candidate/selection policy")
    validate_dataset(root, dataset_id=policy["dataset_id"], validate_only=True)
    dataset = root / "data/processed" / policy["dataset_id"]
    primary = read(dataset / "development_primary.jsonl")
    supplementary = read(dataset / "development_supplementary.jsonl")
    for role, rows in (("primary", primary), ("supplementary", supplementary)):
        if (len(rows) != policy[f"expected_{role}_articles"]
                or any(r["modelling_role"] != role for r in rows)):
            raise ValueError("Unexpected development cohort")
    documents = primary + supplementary
    keys = [(r["source"], r["document_id"]) for r in documents]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate development document")
    group_counts = Counter(r["split_group_id"] for r in primary)
    dominant = min(group_counts, key=lambda g: (-group_counts[g], g))
    audits = [audit_row(row, limit) for limit in limits for row in documents]
    chosen = choose_limit([r for r in audits if r["modelling_role"] == "primary"], limits)
    candidates = [{"candidate_id": candidate_id(limit), "body_word_limit": limit,
                   **{role: stratify([a for a in audits if a["body_word_limit"] == limit
                                     and a["modelling_role"] == role], dominant)
                      for role in ("primary", "supplementary")}} for limit in limits]
    selected_rule = {"rule_version": "nlp_text_input_1", "selected_candidate": candidate_id(chosen),
                     "body_word_limit": chosen, "include_title": True, "separator": "\n\n",
                     "word_definition": policy["word_definition"], "prefix_rule": policy["prefix_rule"],
                     "input_rule": policy["input_rule"], "selection": policy["selection"],
                     "selected_on": "2019-2022 primary development evidence only",
                     "model_tokenizer_truncation_allowed": False,
                     "embedding_adapter_status": "Not implemented; tokenizer-aware chunks and pooling must preserve selected input coverage.",
                     "guide_and_labels_changed": False}
    summary = {"policy_version": policy["policy_version"], "selected_candidate": candidate_id(chosen),
               "largest_primary_component": {"split_group_id": dominant, "articles": group_counts[dominant]},
               "candidates": candidates, "limitations": policy["limitations"],
               "metric_scope": policy["metric_scope"], "evaluation_reference_labels_loaded": False,
               "models_fitted": False, "financial_outcomes_loaded": False,
               "selection_uses_supplementary": False, "excluded_records_audited": False}
    outputs = {"policy_snapshot.json": policy, "selected_rule.json": selected_rule,
               "summary.json": summary, "article_audit.jsonl": audits,
               "report.md": report_markdown(summary)}
    for role, rows in (("primary", primary), ("supplementary", supplementary)):
        outputs[f"inputs_{role}.jsonl"] = [
            {**{k: r[k] for k in ("source", "document_id", "record_version_hash", "body_model_sha256",
                                  "near_family_id", "split_group_id", "modelling_role")},
             "rule_version": selected_rule["rule_version"], **model_input(r["title"], r["body_model"], chosen)}
            for r in rows]
    paths = [root / POLICY, root / "scripts/audit_evidence_retention.py", dataset / "manifest.json",
             dataset / "development_primary.jsonl", dataset / "development_supplementary.jsonl"]
    return outputs, {str(p.relative_to(root)): digest(p) for p in paths}


def run(root=ROOT, audit_id="nlp_input_v1", validate_only=False):
    if not re.fullmatch(r"[a-z0-9_]+", audit_id):
        raise ValueError("Invalid audit ID")
    output = root / "data/audit" / audit_id
    if output.exists() and not validate_only:
        raise ValueError("Existing audit left unchanged; use --validate or a new --audit-id")
    if validate_only and not output.is_dir():
        raise ValueError("Audit does not exist")
    artifacts, inputs = assemble(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".retention_build_", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, value in artifacts.items():
            if name.endswith(".jsonl"):
                write_rows(stage / name, value)
            elif name.endswith(".md"):
                (stage / name).write_text(value, encoding="utf-8")
            else:
                save_json(stage / name, value)
        hashes = {name: digest(stage / name) for name in sorted(artifacts)}
        if any(digest(root / name) != expected for name, expected in inputs.items()):
            raise ValueError("Inputs changed during retention audit")
        if validate_only:
            saved = json.loads((output / "manifest.json").read_text())
            if (saved["audit_id"] != audit_id or saved["schema_version"] != 1
                    or saved["input_sha256"] != inputs or saved["artifact_sha256"] != hashes
                    or any(digest(output / name) != expected for name, expected in hashes.items())):
                raise ValueError("Audit/input drift or non-reproducible artifacts")
        else:
            save_json(stage / "manifest.json", {"audit_id": audit_id, "schema_version": 1,
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "input_sha256": inputs, "artifact_sha256": hashes})
            output.mkdir(exist_ok=False)
            for path in sorted(stage.iterdir()):
                path.rename(output / path.name)
    return {"audit": str(output), "validated": validate_only,
            "selected_rule": artifacts["selected_rule.json"],
            "report": artifacts["report.md"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-id", default="nlp_input_v1")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    result = run(audit_id=args.audit_id, validate_only=args.validate)
    print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
    print(result["report"])
