"""Check frozen AI review evidence and provenance without changing corpus outputs."""

from collections import Counter
import json
from pathlib import Path

try:
    from .prepare_news_review import ROOT, digest, read, protected_families
except ImportError:
    from prepare_news_review import ROOT, digest, read, protected_families


LABELS = ("relevant", "threat", "disruption", "deescalation")


def validate_revision(batch, annotations):
    """Bind current decisions to preserved inputs and an explicit revision ledger."""
    path = batch / "revision_guide_0_2.json"
    if not path.exists():
        if (batch / "annotations.ai_initial.jsonl").exists() and (
                digest(batch / "annotations.jsonl") != digest(batch / "annotations.ai_initial.jsonl")):
            raise ValueError("Changed decisions require a revision ledger")
        return
    revision = json.loads(path.read_text(encoding="utf-8"))
    for field, hash_field in (("previous_snapshot", "previous_sha256"),
                              ("initial_snapshot", "initial_sha256"),
                              ("guide_snapshot", "guide_sha256"),
                              ("current_annotation_file", "current_annotation_sha256")):
        if digest(batch / revision[field]) != revision[hash_field]:
            raise ValueError(f"Revision hash mismatch: {field}")
    previous = read(batch / revision["previous_snapshot"])
    decisions = revision["decisions"]
    if len(previous) != len(annotations) or len(decisions) != len(annotations):
        raise ValueError("Incomplete revision ledger")
    for old, new, decision in zip(previous, annotations, decisions):
        for key in ("document_id", "record_version_hash", "article_number"):
            if not old[key] == new[key] == decision[key]:
                raise ValueError("Revision changed article identity or order")
        before = {k: old[k] for k in LABELS}
        after = {k: new[k] for k in LABELS}
        changed = [k for k in LABELS if old[k] != new[k]]
        if (decision["before"] != before or decision["after"] != after
                or decision["changed_label_fields"] != changed):
            raise ValueError("Revision transition does not match saved decisions")
        if new["guide_version"] != revision["guide_version"] or not decision["explanation"]:
            raise ValueError("Missing revision policy or rationale")
        if new["relevant"] == 0 and (
                any(new[k] != 0 for k in ("threat", "disruption", "deescalation"))
                or new["location"] is not None or new["temporal_focus"] is not None):
            raise ValueError("Irrelevant v0.2 article has inconsistent scoped labels/context")


def validate_batch(batch):
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["frozen_artifact_sha256"].items():
        if digest(batch / name) != expected:
            raise ValueError(f"Frozen input changed: {batch.name}/{name}")
    annotations = read(batch / "annotations.jsonl")
    texts = read(batch / "reviewed_texts.jsonl")
    families = read(batch / "family_snapshot.jsonl")
    if not annotations or len(annotations) != len(texts) or len(texts) != len(families):
        raise ValueError("Missing or extra annotation/text/family rows")
    seen = set()
    for a, d, f in zip(annotations, texts, families):
        for key in ("source", "document_id", "record_version_hash", "body_hash", "exact_group_id", "near_family_id"):
            if a[key] != d[key]:
                raise ValueError(f"Annotation/text {key} mismatch")
        if d["near_family_id"] != f["near_family_id"] or d["exact_group_id"] not in f["exact_group_ids"]:
            raise ValueError("Family snapshot does not match reviewed text")
        if d["near_family_id"] in seen:
            raise ValueError("Duplicate family in batch")
        seen.add(d["near_family_id"])
        if not all("2019-01-01" <= day < "2023-01-01" for day in f["publication_dates"]):
            raise ValueError("Later-period family in development packet")
        if not (a["status"] == "ai_annotated" and a["annotator_type"] == "ai"
                and a["human_reviewed"] is False and a["annotator"] and a["annotated_at"]):
            raise ValueError("Missing AI review provenance")
        for key in ("relevant", "threat", "disruption", "deescalation"):
            if a[key] is not None and (type(a[key]) is not int or a[key] not in (0, 1)):
                raise ValueError(f"Invalid {key} label")
        if a["relevant"] == 0 and (a["threat"] != 0 or a["disruption"] != 0):
            raise ValueError("Irrelevant article has nonzero/unresolved dependent labels")
        if a["relevant"] is None and (a["threat"] == 1 or a["disruption"] == 1):
            raise ValueError("Positive event label requires resolved relevance")
        if not a["rationale"] or not a["evidence_spans"]:
            raise ValueError("Missing rationale or evidence")
        if [span["text"] for span in a["evidence_spans"]] != a["evidence_excerpts"]:
            raise ValueError("Evidence list/span mismatch")
        for span in a["evidence_spans"]:
            start, end = span["start"], span["end"]
            if (span["field"] != "body_clean" or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(d["body_clean"])
                    or d["body_clean"][start:end] != span["text"]):
                raise ValueError("Evidence span does not match frozen input")
    summary = json.loads((batch / "review_summary.json").read_text(encoding="utf-8"))
    if digest(batch / "annotations.ai_initial.jsonl") != summary["annotation_sha256"]:
        raise ValueError("Initial AI decisions changed")
    validate_revision(batch, annotations)
    return annotations


def main(root=ROOT):
    documents = read(root / "data/processed/news_documents.jsonl")
    protected, _ = protected_families(root)
    batches = sorted((root / "data/annotations").glob("news_dev_*"))
    all_rows = []
    for batch in batches:
        all_rows.extend(validate_batch(batch))
    # The earlier specialist batch retains its original frozen packet/manifest.
    old = root / "data/annotations/proquest_dev_001/annotations.jsonl"
    if old.exists():
        previous_rows = read(old)
        validate_revision(old.parent, previous_rows)
        all_rows.extend(previous_rows)
    seen = set()
    for a in all_rows:
        matches = [d for d in documents if d["document_id"] == a["document_id"]
                   and (not a.get("source") or d["source"] == a["source"])]
        if len(matches) != 1 or a["record_version_hash"] not in matches[0]["version_hashes"]:
            raise ValueError("Annotation cannot be linked to current corpus version")
        if matches[0]["near_family_id"] in protected:
            raise ValueError("Development annotation overlaps a protected evaluation/future family")
        for span in a.get("evidence_spans", []):
            body = matches[0]["body_clean"]
            if not (span["field"] == "body_clean" and 0 <= span["start"] < span["end"] <= len(body)
                    and body[span["start"]:span["end"]] == span["text"]):
                raise ValueError("Evidence does not match linked corpus version")
        if matches[0]["near_family_id"] in seen:
            raise ValueError("Repeated family across development batches")
        seen.add(matches[0]["near_family_id"])
    report = {"validated_new_batches": len(batches), "total_annotated_articles": len(all_rows),
              "distinct_detected_families": len(seen),
              "human_reviewed": sum(a.get("human_reviewed") is True for a in all_rows),
              "reference_type": "AI; human review optional by user decision",
              "guide_versions": dict(Counter(a["guide_version"] for a in all_rows)),
              "counts": {k: dict(Counter("unresolved" if a[k] is None else str(a[k]) for a in all_rows))
                         for k in ("relevant", "threat", "disruption", "deescalation")}}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
