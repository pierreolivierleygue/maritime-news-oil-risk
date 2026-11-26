"""Build a versioned, development-only NLP dataset without fitting models.

Full text is retained for the subsequent evidence-retention audit. Reference
annotations, reservation assignments and corpus files are never rewritten.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

try:
    from .prepare_news_review import ROOT, digest, read, write_rows, protected_families
    from .reserve_news_evaluation import validate as validate_reservation, save_json
    from .validate_news_annotations import LABELS, validate_batch, validate_revision
except ImportError:
    from prepare_news_review import ROOT, digest, read, write_rows, protected_families
    from reserve_news_evaluation import validate as validate_reservation, save_json
    from validate_news_annotations import LABELS, validate_batch, validate_revision

POLICY = "config/modelling_dataset_v1.json"
IDENTITY = ("source", "document_id", "body_hash", "exact_group_id", "near_family_id",
            "publication_date", "title")
ARTIFACTS = ("development_all.jsonl", "development_primary.jsonl",
             "development_supplementary.jsonl", "development_excluded.jsonl",
             "related_groups.jsonl", "quality_decisions.jsonl", "summary.json")


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        if key in result:
            raise ValueError(f"Duplicate identity for {fields}: {key}")
        result[key] = row
    return result


def prepare_row(annotation, document, version, assignment, family, decision, policy):
    """Join exact versions and fail closed before applying derived-text decisions."""
    for key in IDENTITY:
        if annotation[key] != document[key] or assignment[key] != document[key]:
            raise ValueError(f"Annotation/corpus/assignment identity mismatch: {key}")
    if (annotation["record_version_hash"] not in document["version_hashes"]
            or assignment["record_version_hash"] != document["record_version_hash"]
            or version["record_version_hash"] != annotation["record_version_hash"]):
        raise ValueError("Reviewed version does not match corpus/source version")
    for key in ("document_id", "body_hash", "publication_date", "title", "body_clean"):
        if version[key] != document[key]:
            raise ValueError(f"Exact source version mismatch: {key}")
    body = document["body_clean"]
    if text_digest(body) != document["body_hash"]:
        raise ValueError("Body hash mismatch")
    if (assignment["nlp_training_eligible"] is not True
            or assignment["partition"] != "development_linked"
            or assignment["evaluation_selected"] or assignment["evaluation_eligible"]):
        raise ValueError("Protected evaluation/future record in development")
    dates = family["publication_dates"]
    if (family["near_family_id"] != document["near_family_id"]
            or document["exact_group_id"] not in family["exact_group_ids"]
            or not dates or document["publication_date"] not in dates
            or not all(policy["start"] <= d < policy["end_exclusive"] for d in dates)):
        raise ValueError("Family outside development period or identity mismatch")
    if not (annotation["status"] == "ai_annotated" and annotation["annotator_type"] == "ai"
            and annotation["human_reviewed"] is False and annotation["annotator"]
            and annotation["annotated_at"] and annotation["guide_version"] == "0.2"
            and annotation["input_scope"] == "full_article" and annotation["rationale"]):
        raise ValueError("Missing or incompatible annotation provenance")
    labels = {k: annotation[k] for k in LABELS}
    for value in labels.values():
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("Invalid label (null must remain null)")
    if labels["relevant"] == 0 and any(labels[k] != 0 for k in LABELS[1:]):
        raise ValueError("Irrelevant article has inconsistent dependent labels")
    if labels["relevant"] is None and any(labels[k] == 1 for k in LABELS[1:]):
        raise ValueError("Positive event label requires relevance")

    role = "primary" if document["source"] in policy["primary_sources"] else "supplementary"
    flags, edits = [], []
    model_body = body
    if decision:
        for key in ("source", "document_id", "record_version_hash", "body_hash"):
            if decision[key] != annotation[key]:
                raise ValueError(f"Quality decision version mismatch: {key}")
        action = decision["action"]
        if action not in ("exclude", "retain_flagged", "remove_metadata_suffix") or not decision["reason"]:
            raise ValueError("Invalid quality decision")
        flags = decision["flags"]
        if action == "exclude":
            role = "excluded"
        elif action == "remove_metadata_suffix":
            start = decision["suffix_start"]
            if (type(start) is not int or not 0 < start < len(body)
                    or not body[start:].startswith(" Sujet ERIC:")
                    or len(body) != decision["original_length"]):
                raise ValueError("Reviewed metadata suffix no longer matches")
            model_body = body[:start]
            edits = [{"operation": "remove_suffix", "original_start": start,
                      "original_end": len(body), "removed_sha256": text_digest(body[start:]),
                      "reason": decision["reason"]}]
    spans = annotation["evidence_spans"]
    if not spans or [s["text"] for s in spans] != annotation["evidence_excerpts"]:
        raise ValueError("Missing or inconsistent evidence")
    derived_spans = []
    for span in spans:
        start, end = span["start"], span["end"]
        if (span["field"] != "body_clean" or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(body) or body[start:end] != span["text"]):
            raise ValueError("Evidence does not match reviewed text")
        if end > len(model_body):
            raise ValueError("Cleanup would remove reference evidence")
        derived_spans.append({**span, "field": "body_model"})
    return {
        **{k: document[k] for k in IDENTITY},
        "record_version_hash": annotation["record_version_hash"],
        "corpus_representative_version_hash": document["record_version_hash"],
        "batch_id": annotation["batch_id"], "article_number": annotation["article_number"],
        "split_group_id": assignment["split_group_id"],
        "reservation_partition": assignment["partition"],
        "nlp_training_eligible": assignment["nlp_training_eligible"],
        "modelling_role": role,
        "primary_training_eligible": role == "primary",
        "target_eligible": {k: role == "primary" and labels[k] is not None for k in LABELS},
        "labels": labels, "annotation": annotation,
        "body_clean": body, "body_model": model_body,
        "body_model_sha256": text_digest(model_body),
        "text_scope": "full_article; truncation not yet selected",
        "evidence_spans_model": derived_spans, "text_edits": edits,
        "quality_flags": sorted(set(document.get("quality_flags", []) + flags)),
        "quality_decision": decision,
        "source_input_file": document["input_file"],
        "source_metadata": {k: version.get(k) for k in (
            "publication_datetime", "publication_date_raw", "date_precision", "record_updated_at",
            "retrieved_at", "imported_at", "archive_available_at", "availability_note",
            "query_version", "web_url", "metadata_raw", "provenance")},
    }


def counts(rows):
    return {"articles": len(rows),
            "families": len({r["near_family_id"] for r in rows}),
            "related_components": len({r["split_group_id"] for r in rows}),
            "source_counts": dict(sorted(Counter(r["source"] for r in rows).items())),
            "year_counts": dict(sorted(Counter(r["publication_date"][:4] for r in rows).items())),
            "labels": {k: dict(sorted(Counter(
                "unresolved" if r["labels"][k] is None else str(r["labels"][k])
                for r in rows).items())) for k in LABELS}}


def assemble(root=ROOT):
    policy_path = root / POLICY
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    reservation = root / "data/evaluation" / policy["reservation_id"]
    validate_reservation(root, policy["reservation_id"])
    protected, reservation_inputs = protected_families(root)
    inputs = {policy_path, *reservation_inputs}
    inputs.update(p for p in reservation.iterdir() if p.is_file())
    reservation_manifest = json.loads((reservation / "manifest.json").read_text())
    inputs.update(root / p for p in reservation_manifest["input_sha256"])
    for path in (Path(__file__), Path(__file__).with_name("prepare_news_review.py"),
                 Path(__file__).with_name("reserve_news_evaluation.py"),
                 Path(__file__).with_name("validate_news_annotations.py")):
        inputs.add(root / path.relative_to(ROOT))

    documents_path = root / "data/processed/news_documents.jsonl"
    families_path = root / "data/processed/news_review_families.jsonl"
    inputs.update((documents_path, families_path))
    documents = unique_index(read(documents_path), ("source", "document_id"))
    families = unique_index(read(families_path), ("near_family_id",))
    assignments = unique_index(read(reservation / "assignments.jsonl"), ("source", "document_id"))
    if set(documents) != set(assignments):
        raise ValueError("Incomplete reservation assignments")
    source_versions = {}
    for name in sorted({d["input_file"] for d in documents.values()}):
        path = root / name
        inputs.add(path)
        source_versions[name] = unique_index(read(path), ("document_id", "record_version_hash"))
    decisions = unique_index(policy["quality_decisions"], ("source", "document_id"))
    used_decisions, seen_families, rows = set(), set(), []
    for batch_id in policy["batches"]:
        batch = root / "data/annotations" / batch_id
        inputs.update(p for p in batch.iterdir() if p.is_file())
        if batch_id.startswith("news_dev_"):
            annotations = validate_batch(batch)
            frozen = unique_index(read(batch / "reviewed_texts.jsonl"), ("source", "document_id"))
        else:
            annotations = read(batch / "annotations.jsonl")
            validate_revision(batch, annotations)
            old_manifest = json.loads((batch / "manifest.json").read_text())
            identities = [{k: a[k] for k in ("document_id", "record_version_hash", "body_hash")}
                          for a in annotations]
            if identities != old_manifest["records"]:
                raise ValueError("Specialist annotation identity differs from original packet")
            frozen = None
        for a in annotations:
            if a["batch_id"] != batch_id:
                raise ValueError("Annotation batch identity mismatch")
            key = a["source"], a["document_id"]
            d = documents[key]
            if d["near_family_id"] in protected:
                raise ValueError("Protected family in development")
            if d["near_family_id"] in seen_families:
                raise ValueError("Repeated family across development batches")
            seen_families.add(d["near_family_id"])
            if frozen is not None and frozen[key] != d:
                raise ValueError("Frozen reviewed text differs from corpus")
            version = source_versions[d["input_file"]][a["document_id"], a["record_version_hash"]]
            decision = decisions.get(key)
            if decision:
                used_decisions.add(key)
            rows.append(prepare_row(a, d, version, assignments[key],
                                    families[d["near_family_id"],], decision, policy))
    if used_decisions != set(decisions):
        raise ValueError("Quality decision does not match a development record")
    if len(rows) != policy["expected_annotations"]:
        raise ValueError("Development inventory changed; version the dataset policy")
    rows.sort(key=lambda r: (r["publication_date"], r["source"], r["document_id"]))
    groups = defaultdict(list)
    for row in rows:
        groups[row["split_group_id"]].append(row)
    group_rows = [{"split_group_id": group, "counts": counts(members),
                   "first_publication_date": min(r["publication_date"] for r in members),
                   "last_publication_date": max(r["publication_date"] for r in members),
                   "members": [{k: r[k] for k in ("source", "document_id", "near_family_id",
                                                  "publication_date", "modelling_role")}
                               for r in members]}
                  for group, members in sorted(groups.items())]
    summary = {"policy_version": policy["policy_version"], "all": counts(rows),
               **{role: counts([r for r in rows if r["modelling_role"] == role])
                  for role in ("primary", "supplementary", "excluded")},
               "cleaned_articles": sum(bool(r["text_edits"]) for r in rows),
               "reviewed_quality_decisions": len(decisions),
               "all_evidence_preserved_after_cleanup": True,
               "evaluation_labels_loaded": False,
               "models_fitted": False, "truncation_selected": False,
               "folds_assigned": False,
               "limitations": policy["limitations"]}
    artifacts = {"development_all.jsonl": rows, "related_groups.jsonl": group_rows,
                 "quality_decisions.jsonl": policy["quality_decisions"], "summary.json": summary}
    for role in ("primary", "supplementary", "excluded"):
        artifacts[f"development_{role}.jsonl"] = [r for r in rows if r["modelling_role"] == role]
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted(inputs)}
    return artifacts, hashes


def materialize(path, artifacts):
    for name in ARTIFACTS:
        if name.endswith(".jsonl"):
            write_rows(path / name, artifacts[name])
        else:
            save_json(path / name, artifacts[name])


def run(root=ROOT, dataset_id="nlp_dev_v1", validate_only=False):
    if not re.fullmatch(r"[a-z0-9_]+", dataset_id):
        raise ValueError("Invalid dataset ID")
    output = root / "data/processed" / dataset_id
    if output.exists() and not validate_only:
        raise ValueError("Existing dataset left unchanged; use --validate or a new --dataset-id")
    if validate_only and not output.is_dir():
        raise ValueError("Dataset does not exist")
    artifacts, hashes = assemble(root)
    with tempfile.TemporaryDirectory(prefix=".nlp_build_", dir=output.parent) as temporary:
        stage = Path(temporary)
        materialize(stage, artifacts)
        output_hashes = {name: digest(stage / name) for name in ARTIFACTS}
        manifest = {"dataset_id": dataset_id, "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "input_sha256": hashes, "artifact_sha256": output_hashes}
        # Recheck inputs before publishing, so concurrent source changes fail closed.
        if any(digest(root / name) != expected for name, expected in hashes.items()):
            raise ValueError("Inputs changed during build")
        if validate_only:
            saved = json.loads((output / "manifest.json").read_text())
            if (saved["dataset_id"] != dataset_id or saved["schema_version"] != 1
                    or saved["input_sha256"] != hashes or saved["artifact_sha256"] != output_hashes
                    or any(digest(output / name) != expected for name, expected in output_hashes.items())):
                raise ValueError("Dataset/input drift or non-reproducible artifacts")
        else:
            save_json(stage / "manifest.json", manifest)
            # The target must still be absent; never replace an existing snapshot.
            output.mkdir(exist_ok=False)
            for path in sorted(stage.iterdir()):
                path.rename(output / path.name)
    return {"dataset": str(output), "validated": validate_only, **artifacts["summary.json"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="nlp_dev_v1")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(dataset_id=args.dataset_id, validate_only=args.validate), indent=2))
