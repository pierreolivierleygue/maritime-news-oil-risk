"""Validate held-out AI references without fitting, scoring or changing the reservation."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

try:
    from .prepare_news_review import ROOT, digest, read
    from .reserve_news_evaluation import RESERVATION, validate as validate_reservation
except ImportError:
    from prepare_news_review import ROOT, digest, read
    from reserve_news_evaluation import RESERVATION, validate as validate_reservation


LABELS = ("relevant", "threat", "disruption", "deescalation")
IDENTITY = ("source", "document_id", "record_version_hash", "body_hash",
            "exact_group_id", "near_family_id", "publication_date", "title")
LOCATIONS = {"Hormuz", "Red Sea–Bab el-Mandeb", "both", "uncertain", None}
TEMPORAL = {"current development", "historical background", "mixed", "uncertain", None}


def aware_time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Review/freeze timestamps require a timezone")
    return result


def validate_rows(annotations, queue, texts, freeze):
    """Check identity, holdout status, policy, provenance and literal evidence."""
    if not queue or len(annotations) != len(queue) or len(texts) != len(queue):
        raise ValueError("Missing or extra evaluation rows")
    by_id = {(d["source"], d["document_id"]): d for d in texts}
    if len(by_id) != len(texts):
        raise ValueError("Repeated evaluation text identity")
    seen = set()
    for n, (a, q) in enumerate(zip(annotations, queue), 1):
        d = by_id.get((q["source"], q["document_id"]))
        if d is None or any(a[k] != q[k] or a[k] != d[k] for k in IDENTITY):
            raise ValueError("Evaluation identity/version/order mismatch")
        if a["article_number"] != n or q["article_number"] != n:
            raise ValueError("Evaluation article order changed")
        if a["near_family_id"] in seen:
            raise ValueError("Repeated evaluation family")
        seen.add(a["near_family_id"])
        if (q["status"] != "reserved_unlabelled"
                or any(k in q for k in LABELS)
                or not "2023-01-01" <= a["publication_date"] < "2024-01-01"):
            raise ValueError("Reservation is not an unlabelled 2023 queue")
        for row in (a, q):
            if (row["nlp_training_eligible"] is not False
                    or row["partition"] != "evaluation_reserved"
                    or row["evaluation_selected"] is not True
                    or row["evaluation_eligible"] is not True):
                raise ValueError("Evaluation training/selection contamination")
        if a["split_group_id"] != q["split_group_id"]:
            raise ValueError("Evaluation grouping changed")
        if (a["guide_version"] != freeze["guide_version"]
                or a["guide_sha256"] != freeze["guide_sha256"]
                or a["reservation_id"] != freeze["reservation_id"]):
            raise ValueError("Wrong frozen guide/reservation")
        if (a["status"] != "ai_annotated" or a["annotator_type"] != "ai"
                or a["human_reviewed"] is not False or not a["annotator"]
                or a["input_scope"] != "full_article"
                or aware_time(a["annotated_at"]) <= aware_time(freeze["guide_frozen_at"])):
            raise ValueError("Missing or invalid AI review provenance")
        if hashlib.sha256(d["body_clean"].encode("utf-8")).hexdigest() != a["body_hash"]:
            raise ValueError("Reviewed body hash mismatch")
        for k in LABELS:
            if a[k] is not None and (type(a[k]) is not int or a[k] not in (0, 1)):
                raise ValueError(f"Invalid {k} label")
        if a["location"] not in LOCATIONS or a["temporal_focus"] not in TEMPORAL:
            raise ValueError("Invalid context")
        if a["relevant"] == 0 and (any(a[k] != 0 for k in LABELS[1:])
                or a["location"] is not None or a["temporal_focus"] is not None):
            raise ValueError("Irrelevant article has dependent labels/context")
        if a["relevant"] is None and any(a[k] == 1 for k in LABELS[1:]):
            raise ValueError("Positive target requires resolved relevance")
        if a["relevant"] == 1 and (a["location"] is None or a["temporal_focus"] is None):
            raise ValueError("Relevant article lacks context")
        if not a["rationale"] or not a["evidence_spans"]:
            raise ValueError("Missing rationale/evidence")
        if any(a[k] is None for k in LABELS) and not a.get("review_notes"):
            raise ValueError("Unresolved target requires an uncertainty note")
        if [s["text"] for s in a["evidence_spans"]] != a["evidence_excerpts"]:
            raise ValueError("Evidence list/span mismatch")
        for s in a["evidence_spans"]:
            start, end = s["start"], s["end"]
            if (s["field"] != "body_clean" or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(d["body_clean"])
                    or d["body_clean"][start:end] != s["text"]):
                raise ValueError("Evidence span does not match frozen text")


def label_counts(rows):
    return {k: {name: sum(a[k] == value if value is not None else a[k] is None
                          for a in rows)
                for name, value in (("yes", 1), ("no", 0), ("unresolved", None))}
            for k in LABELS}


def summarize(rows):
    groups = Counter(a["split_group_id"] for a in rows)
    dominant = sorted(groups, key=lambda g: (-groups[g], g))[0]
    outside = [a for a in rows if a["split_group_id"] != dominant]
    return dict(
        total_annotated_articles=len(rows),
        distinct_detected_families=len({a["near_family_id"] for a in rows}),
        related_report_components=len(groups),
        source_counts=dict(Counter(a["source"] for a in rows)),
        counts=label_counts(rows),
        # Resolved labels stay usable independently of other null targets.
        resolved_by_target={k: sum(a[k] is not None for a in rows) for k in LABELS},
        component_counts={g: dict(articles=groups[g],
                              counts=label_counts([a for a in rows if a["split_group_id"] == g]))
                          for g in sorted(groups)},
        dominant_component=dict(split_group_id=dominant, articles=groups[dominant]),
        excluding_dominant_component=dict(articles=len(outside), counts=label_counts(outside)),
        human_reviewed=sum(a["human_reviewed"] is True for a in rows),
        reference_type="AI reference labels; not independently established event accuracy",
    )


def verify_hashes(base, expected):
    for name, value in expected.items():
        path = base / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe artifact path")
        if not path.is_file() or digest(path) != value:
            raise ValueError(f"Evaluation labelling hash mismatch: {name}")


def validate(root=ROOT, reservation_id=RESERVATION):
    reservation = validate_reservation(root, reservation_id)
    batch = root / "data/evaluation" / reservation_id
    review = batch / "labelling_v1"
    manifest = json.loads((review / "manifest.json").read_text())
    frozen = json.loads((batch / "manifest.json").read_text())
    required_inputs = set(frozen["frozen_artifact_sha256"]) | {"manifest.json"}
    if set(manifest["reservation_artifact_sha256"]) != required_inputs:
        raise ValueError("Incomplete reservation input binding")
    verify_hashes(batch, manifest["reservation_artifact_sha256"])
    required_outputs = {"annotations.jsonl", "annotations.ai_initial.jsonl", "review_summary.json",
                        "annotation_review.md", "related_report_review.json"}
    required_outputs.update(f"decision_{n:03}.json" for n in range(1, len(read(batch / "queue_snapshot.jsonl")) + 1))
    if set(manifest["artifact_sha256"]) != required_outputs:
        raise ValueError("Incomplete labelling artifact binding")
    verify_hashes(review, manifest["artifact_sha256"])
    rows = read(review / "annotations.jsonl")
    if rows != read(review / "annotations.ai_initial.jsonl"):
        raise ValueError("Initial references changed; use a separately versioned review")
    decisions = [json.loads((review / f"decision_{n:03}.json").read_text())
                 for n in range(1, len(rows) + 1)]
    if rows != decisions:
        raise ValueError("References differ from saved individual decisions")
    validate_rows(rows, read(batch / "queue_snapshot.jsonl"), read(batch / "evaluation_texts.jsonl"),
                  json.loads((batch / "guide_freeze.json").read_text()))
    summary = summarize(rows)
    if json.loads((review / "review_summary.json").read_text()) != summary:
        raise ValueError("Summary differs from references")
    if (manifest["status"] != "ai_annotated" or manifest["reservation_id"] != reservation_id
            or manifest["guide_sha256"] != reservation["guide_sha256"]
            or manifest["evaluation_articles"] != len(rows)):
        raise ValueError("Labelling manifest identity/status mismatch")
    return dict(reservation_id=reservation_id, status="validated_ai_annotations",
                total_annotated_articles=len(rows), counts=summary["counts"],
                resolved_by_target=summary["resolved_by_target"],
                related_report_components=summary["related_report_components"],
                largest_component_articles=summary["dominant_component"]["articles"],
                original_reservation_and_prior_data_preserved=True,
                validation_scope="Integrity and consistency; not independent semantic validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(validate(), indent=2))
