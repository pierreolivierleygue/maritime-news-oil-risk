"""Build immutable retrospective news features and next-week OVX targets; no fitting."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, time, timedelta, timezone
import json
import math
from pathlib import Path
import platform
import tempfile

try:
    from .audit_evidence_retention import model_input
    from .nlp_features import keyword_predictions
    from .prepare_modelling_dataset import text_digest
    from .prepare_news_review import ROOT, digest, read, write_rows
    from .reserve_news_evaluation import save_json
    from .validate_evaluation_annotations import verify_hashes
except ImportError:
    from audit_evidence_retention import model_input
    from nlp_features import keyword_predictions
    from prepare_modelling_dataset import text_digest
    from prepare_news_review import ROOT, digest, read, write_rows
    from reserve_news_evaluation import save_json
    from validate_evaluation_annotations import verify_hashes

POLICY = "config/weekly_finance_v1.json"
TARGETS = ("relevant", "threat", "disruption")
FT, GUARDIAN = "Financial Times; London (UK)", "The Guardian"
DAY, WEEK = timedelta(days=1), timedelta(days=7)


def monday(day):
    return day - timedelta(days=day.weekday())


def dates(first, last):
    while first <= last:
        yield first
        first += DAY


def weeks(policy):
    first, last = monday(date.fromisoformat(policy["start"])), date.fromisoformat(policy["end"])
    while first <= last:
        yield first
        first += WEEK


def timestamp(document):
    raw = document.get("publication_datetime")
    if raw:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if value.tzinfo is None:
            raise ValueError("Publication timestamp lacks timezone")
        value = value.astimezone(timezone.utc)
        if value.date().isoformat() != document["publication_date"]:
            raise ValueError("Publication date and UTC timestamp disagree")
        return value, "published_timestamp_utc"
    return datetime.combine(date.fromisoformat(document["publication_date"]),
                            time(23, 59, 59), timezone.utc), "date_only_end_of_day_utc_assumption"


def load_inputs(root, policy):
    """Verify frozen code/data chains without reading annotations as predictors."""
    verify_hashes(root, policy["frozen_sha256"])
    dependencies = dict(policy["frozen_sha256"])
    for folder in ("data/experiments/nlp_dev_comparison_v1", "data/audit/nlp_input_v1",
                   "data/processed/nlp_dev_v1"):
        manifest = json.loads((root / folder / "manifest.json").read_text())
        verify_hashes(root, manifest["input_sha256"])
        verify_hashes(root / folder, manifest["artifact_sha256"])
        dependencies.update(manifest["input_sha256"])
        dependencies.update({f"{folder}/{k}": v for k, v in manifest["artifact_sha256"].items()})
    reservation = root / "data/evaluation/nlp_eval_2023_v1"
    manifest = json.loads((reservation / "manifest.json").read_text())
    verify_hashes(root, manifest["input_sha256"])
    verify_hashes(reservation, manifest["frozen_artifact_sha256"])
    dependencies.update(manifest["input_sha256"])
    dependencies.update({str(reservation.relative_to(root) / k): v
                         for k, v in manifest["frozen_artifact_sha256"].items()})
    dev = root / "data/experiments/nlp_dev_comparison_v1"
    selection = json.loads((dev / "selection.json").read_text())
    protocol = json.loads((dev / "protocol.json").read_text())
    rule = json.loads((root / "data/audit/nlp_input_v1/selected_rule.json").read_text())
    if (selection["method"] != policy["primary_method"] or selection["method"] != "keyword"
            or rule["selected_candidate"] != "full" or rule["body_word_limit"] is not None
            or rule["include_title"] is not True or rule["separator"] != "\n\n"):
        raise ValueError("Frozen keyword/full-text selection changed")
    for name in (POLICY, "scripts/build_weekly_finance.py", "scripts/nlp_features.py",
                 "scripts/audit_evidence_retention.py", "scripts/prepare_modelling_dataset.py",
                 "scripts/prepare_news_review.py", "scripts/reserve_news_evaluation.py",
                 "scripts/validate_evaluation_annotations.py"):
        dependencies[name] = digest(root / name)
    return protocol, json.loads((root / "config/modelling_dataset_v1.json").read_text()), dependencies


def classify_documents(documents, assignments, quality, keyword, policy):
    assigned = {(r["source"], r["document_id"]): r for r in assignments}
    keys = [(r["source"], r["document_id"]) for r in documents]
    if len(set(keys)) != len(keys) or len(assigned) != len(assignments) or set(keys) != set(assigned):
        raise ValueError("Corpus/assignment identities differ")
    decisions = {(r["source"], r["document_id"]): r for r in quality["quality_decisions"]}
    seen, predictions, exclusions = set(), [], []
    for d in sorted(documents, key=lambda r: (r["publication_date"], r["source"], r["document_id"])):
        key = d["source"], d["document_id"]
        a = assigned[key]
        for field in ("record_version_hash", "body_hash", "exact_group_id", "near_family_id",
                      "publication_date", "title"):
            if d[field] != a[field]:
                raise ValueError(f"Corpus/assignment mismatch: {field}")
        if text_digest(d["body_clean"]) != d["body_hash"]:
            raise ValueError("Article body hash mismatch")
        if d["exact_group_id"] != text_digest(d["source"] + "\n" + d["body_hash"]):
            raise ValueError("Exact group is not source/body identity")
        decision = decisions.get(key)
        if decision:
            seen.add(key)
            if (decision["record_version_hash"] not in d["version_hashes"]
                    or decision["body_hash"] != d["body_hash"]):
                raise ValueError("Stale development quality decision")
        reason = ("supplementary_source" if d["source"] not in policy["sources"] else
                  "outside_period" if not policy["start"] <= d["publication_date"] <= policy["end"] else
                  "development_quality_exclusion" if decision and decision["action"] == "exclude" else None)
        base = {k: d[k] for k in ("source", "document_id", "record_version_hash", "body_hash",
                                  "exact_group_id", "near_family_id", "publication_date", "title")}
        if reason:
            exclusions.append({**base, "reason": reason, "quality_decision": decision})
            continue
        body = d["body_clean"]
        if decision and decision["action"] == "remove_metadata_suffix":
            start = decision["suffix_start"]
            if len(body) != decision["original_length"] or not body[start:].startswith(" Sujet ERIC:"):
                raise ValueError("Verified metadata suffix changed")
            body = body[:start]
        pub, basis = timestamp(d)
        inp = model_input(d["title"], body, None)
        predictions.append({**base, "split_group_id": a["split_group_id"],
            "publication_proxy_at": pub.isoformat(), "timestamp_basis": basis,
            "body_model_sha256": text_digest(body), "input_sha256": inp["input_sha256"],
            "predictions": keyword_predictions(inp["input_text"], keyword),
            "quality_action": decision["action"] if decision else "retain",
            "quality_flags": sorted(set(d.get("quality_flags", []) + (decision["flags"] if decision else []))),
            "source_url": d.get("web_url") or f"https://www.proquest.com/docview/{d['document_id']}",
            "record_updated_at": d.get("record_updated_at"),
            "archive_available_at": d.get("archive_available_at"),
            "historical_availability_verified": False})
    if seen != set(decisions):
        raise ValueError("Missing development quality decision identities")
    return predictions, exclusions


def coverage(source, first, last, policy, ft_years, guardian_months):
    in_range = date.fromisoformat(policy["start"]) <= first <= last <= date.fromisoformat(policy["end"])
    if source == GUARDIAN:
        complete = {d.isoformat()[:7] for d in dates(first, last)} <= guardian_months
        status = "query_collection_complete" if complete else "collection_unverified"
        accepted = complete
    else:
        statuses = {ft_years.get(d.year, "unverified") for d in dates(first, last)}
        accepted = statuses <= {"counts_reconciled", "partial_pending_search_reconciliation"}
        status = ("annual_counts_reconciled_filters_unverified" if statuses == {"counts_reconciled"}
                  else "known_annual_gap_accepted" if accepted else "collection_unverified")
        complete = False  # Annual reconciliation does not establish complete query retrieval.
    return {"source": source, "publication_window_start": first.isoformat(),
            "publication_window_end": last.isoformat(), "window_in_corpus_range": in_range,
            "status": status, "complete_query_collection": complete and in_range,
            "accepted_observed_coverage": accepted and in_range}


def aggregate_news(predictions, policy, ft_years, guardian_months):
    weekly, members, overflow = [], [], []
    week_set = set(weeks(policy))
    for timing, setting in policy["timing"].items():
        buckets = defaultdict(list)
        for row in predictions:
            effective = datetime.fromisoformat(row["publication_proxy_at"]) + timedelta(
                days=setting["news_delay_days"][row["source"]])
            item = {**row, "effective_at": effective.isoformat()}
            week = monday(effective.date())
            if week not in week_set:
                overflow.append({"timing": timing, "source": row["source"], "document_id": row["document_id"],
                                 "effective_at": effective.isoformat(), "reason": "shifted_beyond_output_range"})
            else:
                buckets[row["source"], week].append(item)
        for week in weeks(policy):
            source_groups = {}
            for source in policy["sources"]:
                grouped = defaultdict(list)
                for row in buckets[source, week]:
                    grouped[row["exact_group_id"]].append(row)
                representatives = []
                for group, copies in sorted(grouped.items()):
                    copies.sort(key=lambda r: (r["effective_at"], r["document_id"], r["record_version_hash"]))
                    chosen = copies[0]
                    representatives.append(chosen)
                    for row in copies:
                        members.append({"timing": timing, "week_start": week.isoformat(), "source": source,
                            "document_id": row["document_id"], "record_version_hash": row["record_version_hash"],
                            "exact_group_id": group, "effective_at": row["effective_at"],
                            "representative_document_id": chosen["document_id"],
                            "counted": row is chosen,
                            "disagrees_with_representative": row["predictions"] != chosen["predictions"]})
                source_groups[source] = representatives
            for view, sources in policy["views"].items():
                reps = [r for source in sources for r in source_groups[source]]
                coverage_rows = [coverage(source, week - timedelta(days=setting["news_delay_days"][source]),
                    week + 6 * DAY - timedelta(days=setting["news_delay_days"][source]),
                    policy, ft_years, guardian_months) for source in sources]
                available = all(r["accepted_observed_coverage"] for r in coverage_rows)
                complete = all(r["complete_query_collection"] for r in coverage_rows)
                n = len(reps)
                counts = {t: sum(r["predictions"][t] for r in reps) for t in TARGETS}
                raw_shares = {t: counts[t] / n if n else None for t in ("threat", "disruption")}
                weekly.append({"timing": timing, "view": view, "week_start": week.isoformat(),
                    "week_end": (week + 6 * DAY).isoformat(),
                    "cutoff_exclusive_utc": datetime.combine(week + WEEK, time(), timezone.utc).isoformat(),
                    "observed_documents": sum(len(buckets[s, week]) for s in sources),
                    "candidate_count": n, "counts": counts,
                    "exact_copies_removed": sum(len(buckets[s, week]) for s in sources) - n,
                    "near_family_count": len({r["near_family_id"] for r in reps}),
                    "related_component_count": len({r["split_group_id"] for r in reps}),
                    "candidate_empty": n == 0, "raw_shares": raw_shares,
                    "features": {"news_volume": math.log1p(counts["relevant"]) if available else None,
                        **{f"{t}_share": (raw_shares[t] if n else 0.0) if available else None
                           for t in raw_shares}},
                    "news_features_available": available, "coverage": coverage_rows,
                    "zero_interpretation": ("observed_records" if n else "no_eligible_query_matches" if complete
                                            else "no_observed_records_completeness_unverified"),
                    "members": [{k: r[k] for k in ("source", "document_id", "exact_group_id", "near_family_id",
                                      "split_group_id", "effective_at", "source_url")} for r in reps]})
    return weekly, members, overflow


def read_ovx(path, policy):
    values, previous = {}, None
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["observation_date", "OVXCLS"]:
            raise ValueError("Unexpected OVX columns")
        for row in reader:
            day = date.fromisoformat(row["observation_date"])
            if previous is not None and day <= previous:
                raise ValueError("Duplicate or unsorted OVX date")
            previous = day
            if not policy["start"] <= day.isoformat() <= policy["end"]:
                continue
            if day.weekday() > 4:
                raise ValueError("Unexpected weekend OVX observation")
            raw = row["OVXCLS"].strip()
            value = None if raw in ("", ".") else float(raw)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Non-finite or nonpositive OVX value")
            values[day] = value
    if not values:
        raise ValueError("Empty OVX series")
    return values


def aggregate_ovx(values, policy):
    rows = []
    for week in weeks(policy):
        weekdays = list(dates(week, week + 4 * DAY))
        numeric = [(d, values[d]) for d in weekdays if values.get(d) is not None]
        blank = [d.isoformat() for d in weekdays if d in values and values[d] is None]
        absent = [d.isoformat() for d in weekdays if d not in values]
        full = policy["start"] <= week.isoformat() and (week + 4 * DAY).isoformat() <= policy["end"]
        rows.append({"week_start": week.isoformat(), "week_end": (week + 4 * DAY).isoformat(),
            "observed_mean": sum(v for _, v in numeric) / len(numeric) if numeric else None,
            "observations": len(numeric), "observation_dates": [d.isoformat() for d, _ in numeric],
            "explicit_missing_dates": blank, "absent_weekdays": absent,
            "full_window_in_range": full,
            "eligible": full and len(numeric) >= policy["minimum_ovx_observations"],
            "exchange_calendar_verified": False, "release_time_verified": False,
            "imputed_observations": 0})
    return rows


def split_for_target(start, end, policy):
    for name, (first, last) in policy["splits"].items():
        if first <= start.isoformat() <= end.isoformat() <= last:
            return name
    return None


def align_financial(news, market, policy):
    market = {r["week_start"]: r for r in market}
    rows = []
    for n in news:
        week = date.fromisoformat(n["week_start"])
        target_start, target_end = week + WEEK, week + WEEK + 4 * DAY
        lag = policy["timing"][n["timing"]]["ovx_lag_weeks"]
        history_starts = [week - (lag + i) * WEEK for i in range(policy["ovx_history_weeks"])]
        history = [market.get(w.isoformat()) for w in history_starts]
        history_ok = all(r is not None and r["eligible"] for r in history)
        target = market.get(target_start.isoformat())
        split = split_for_target(target_start, target_end, policy)
        reasons = []
        if not n["news_features_available"]:
            reasons.append("incomplete_or_unverified_news_window")
        if not history_ok:
            reasons.append("insufficient_or_partial_ovx_history")
        if not target or not target["eligible"]:
            reasons.append("insufficient_or_partial_ovx_target")
        if split is None:
            reasons.append("target_outside_split_or_crossing_boundary")
        rows.append({"timing": n["timing"], "view": n["view"], "feature_week_start": week.isoformat(),
            "cutoff_exclusive_utc": n["cutoff_exclusive_utc"],
            "target_week_start": target_start.isoformat(), "target_week_end": target_end.isoformat(),
            "target_mature_after_utc": datetime.combine(target_end + DAY, time(), timezone.utc).isoformat(),
            "split": split, "eligible": not reasons, "exclusion_reasons": reasons,
            "features": {"ovx_latest": history[0]["observed_mean"] if history[0] and history[0]["eligible"] else None,
                "ovx_trailing_four_week_mean": sum(r["observed_mean"] for r in history) / len(history) if history_ok else None,
                **n["features"]},
            "target_ovx_mean": target["observed_mean"] if target and target["eligible"] else None,
            "target_observations": target["observations"] if target else 0,
            "ovx_history_weeks": [w.isoformat() for w in history_starts],
            "ovx_history_observations": [r["observations"] if r else 0 for r in history],
            "candidate_count": n["candidate_count"], "candidate_empty": n["candidate_empty"],
            "coverage": n["coverage"], "zero_interpretation": n["zero_interpretation"],
            "market_calendar_verified": False, "real_time_eligibility_verified": False})
    return rows


def summarize(predictions, excluded, news, market, financial, members, overflow, policy):
    result = {"version": policy["version"], "primary_method": policy["primary_method"],
        "primary_timing": policy["primary_timing"], "primary_view": policy["primary_view"],
        "classified_documents": len(predictions), "source_documents": dict(Counter(r["source"] for r in predictions)),
        "excluded_documents": dict(Counter(r["reason"] for r in excluded)),
        "cleaned_documents": sum(r["quality_action"] == "remove_metadata_suffix" for r in predictions),
        "weekly_market_rows": len(market), "market_numeric_observations": sum(r["observations"] for r in market),
        "market_explicit_missing": sum(len(r["explicit_missing_dates"]) for r in market),
        "minimum_weekly_observations": min(r["observations"] for r in market),
        "shifted_outside_range": dict(Counter(r["timing"] for r in overflow)),
        "prediction_disagreements_with_exact_representative": dict(Counter(r["timing"] for r in members if r["disagrees_with_representative"])),
        "panels": {}, "financial_models_fitted": False, "financial_scores_computed": False,
        "reference_labels_used_as_predictions": False, "limitations": policy["limitations"]}
    for timing in policy["timing"]:
        for view in policy["views"]:
            nr = [r for r in news if (r["timing"], r["view"]) == (timing, view)]
            fr = [r for r in financial if (r["timing"], r["view"]) == (timing, view)]
            result["panels"][f"{timing}/{view}"] = {
                "weeks": len(nr), "eligible_rows": sum(r["eligible"] for r in fr),
                "eligible_by_split": dict(Counter(r["split"] for r in fr if r["eligible"])),
                "exclusion_reasons": dict(Counter(reason for r in fr for reason in r["exclusion_reasons"])),
                "candidate_text_weeks": sum(r["candidate_count"] for r in nr),
                "exact_copies_removed": sum(r["exact_copies_removed"] for r in nr),
                "weeks_with_observed_candidates": sum(r["candidate_count"] > 0 for r in nr),
                "weeks_with_predicted_relevance": sum(r["counts"]["relevant"] > 0 for r in nr),
                "counts": {t: sum(r["counts"][t] for r in nr) for t in TARGETS}}
    return result


def compute(stage, root, policy):
    protocol, quality, dependencies = load_inputs(root, policy)
    predictions, excluded = classify_documents(read(root / "data/processed/news_documents.jsonl"),
        read(root / "data/evaluation/nlp_eval_2023_v1/assignments.jsonl"), quality, protocol["keyword"], policy)
    ft = json.loads((root / "data/audit/proquest_ft_collection.json").read_text())
    guardian = json.loads((root / "data/audit/guardian_retrieval.json").read_text())
    news, members, overflow = aggregate_news(predictions, policy,
        {r["year"]: r["status"] for r in ft["annual"]},
        {r["month"] for r in guardian["periods"] if r["status"] in ("complete", "confirmed_zero_results")})
    # News features are fixed and saved before reading any outcome values.
    for name, rows in (("article_predictions", predictions), ("excluded_documents", excluded),
                       ("weekly_news", news), ("deduplication_members", members), ("shifted_outside_range", overflow)):
        write_rows(stage / f"{name}.jsonl", rows)
    save_json(stage / "news_commit.json", {"financial_outcomes_loaded": False,
        "article_predictions_sha256": digest(stage / "article_predictions.jsonl"),
        "weekly_news_sha256": digest(stage / "weekly_news.jsonl"), "primary_method": "keyword"})
    values = read_ovx(root / "data/raw/ovx_2019_2025.csv", policy)
    market = aggregate_ovx(values, policy)
    financial = align_financial(news, market, policy)
    summary = summarize(predictions, excluded, news, market, financial, members, overflow, policy)
    for name, rows in (("weekly_ovx", market), ("financial_rows", financial)):
        write_rows(stage / f"{name}.jsonl", rows)
    for split in policy["splits"]:
        write_rows(stage / f"primary_{split}.jsonl", [r for r in financial if r["eligible"] and r["split"] == split
            and r["timing"] == policy["primary_timing"] and r["view"] == policy["primary_view"]])
    save_json(stage / "protocol.json", policy)
    save_json(stage / "summary.json", summary)
    return summary, dependencies


def run(root=ROOT, validate_only=False):
    policy = json.loads((root / POLICY).read_text())
    output = root / policy["output_directory"]
    if not output.resolve().is_relative_to((root / "data/processed").resolve()):
        raise ValueError("Unsafe output directory")
    if output.exists() and not validate_only:
        raise ValueError("Weekly dataset exists; use --validate to reproduce without overwriting")
    saved = None
    if validate_only:
        saved = json.loads((output / "manifest.json").read_text())
        verify_hashes(root, saved["input_sha256"])
        verify_hashes(output, saved["artifact_sha256"])
    initial_policy = digest(root / POLICY)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".weekly_finance_", dir=output.parent) as temporary:
        stage = Path(temporary)
        summary, inputs = compute(stage, root, policy)
        outputs = {p.name: digest(p) for p in sorted(stage.iterdir()) if p.is_file()}
        verify_hashes(root, inputs)
        if digest(root / POLICY) != initial_policy:
            raise ValueError("Policy changed during computation")
        runtime = {"python": platform.python_version(), "machine": platform.machine()}
        if validate_only:
            if (saved["input_sha256"] != inputs or saved["artifact_sha256"] != outputs
                    or saved["runtime"] != runtime or saved["schema_version"] != 1):
                raise ValueError("Weekly dataset does not reproduce exactly")
        else:
            save_json(stage / "manifest.json", {"schema_version": 1, "status": "completed_retrospective_dataset",
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
    print(json.dumps(run(validate_only=args.validate), indent=2))
