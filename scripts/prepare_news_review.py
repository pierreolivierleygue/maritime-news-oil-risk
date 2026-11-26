"""Freeze private early-period review packets; never overwrite saved annotations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
GUIDE_VERSION = "0.2"


def read(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def protected_families(root=ROOT):
    """Fail closed on stale reservations, including holds with no labels yet."""
    protected, inputs = set(), []
    for batch in sorted((root / "data/evaluation").glob("*")):
        if not batch.is_dir():
            continue
        manifest_path = batch / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Incomplete evaluation reservation: {batch.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs.append(manifest_path)
        for name, expected in manifest["frozen_artifact_sha256"].items():
            if digest(batch / name) != expected:
                raise ValueError(f"Frozen evaluation artifact changed: {name}")
        for name in ("data/processed/news_documents.jsonl", "data/processed/news_review_families.jsonl"):
            if digest(root / name) != manifest["input_sha256"][name]:
                raise ValueError("Evaluation corpus changed; re-audit the reservation before development")
        assignments = batch / "assignments.jsonl"
        inputs.append(assignments)
        protected.update(a["near_family_id"] for a in read(assignments) if not a["nlp_training_eligible"])
    return protected, inputs


def prepare(batch_id, mode, root=ROOT):
    if not re.fullmatch(r"[a-z0-9_]+", batch_id):
        raise ValueError("Batch ID must contain only lowercase letters, digits and underscores.")
    batch = root / "data/annotations" / batch_id
    if batch.exists():
        raise ValueError(f"Existing batch left unchanged: {batch}")
    doc_path = root / "data/processed/news_documents.jsonl"
    family_path = root / "data/processed/news_review_families.jsonl"
    documents = read(doc_path)
    by_key = {(d["source"], d["document_id"]): d for d in documents}
    families = {f["near_family_id"]: f for f in read(family_path)}
    inputs = [doc_path, family_path, root / "ANNOTATION_GUIDE.md"]
    excluded, reservation_inputs = protected_families(root)
    inputs.extend(reservation_inputs)
    # Protect completed reviews and saved, still-unlabelled development packets.
    for path in sorted((root / "data/annotations").glob("*/*.jsonl")):
        if path.name not in ("annotations.jsonl", "queue_snapshot.jsonl"):
            continue
        inputs.append(path)
        for annotation in read(path):
            matches = [d for d in documents if d["document_id"] == annotation["document_id"]
                       and (not annotation.get("source") or d["source"] == annotation["source"])]
            if len(matches) != 1 or annotation["record_version_hash"] not in matches[0]["version_hashes"]:
                raise ValueError("Existing annotation cannot be uniquely linked to current corpus.")
            excluded.add(matches[0]["near_family_id"])
    if mode == "queue":
        queue_path = root / "data/processed/news_screening_review_queue.jsonl"
        inputs.append(queue_path)
        queue = read(queue_path)
        selection = "Original review_v1 queue, three matches and three nonmatches per source; development only."
    elif mode == "expand":
        used = set(excluded)
        queue = []
        for year in ("2020", "2021", "2022"):
            for source in ("Financial Times; London (UK)", "The Guardian"):
                for candidate in (True, False):
                    pool = [d for d in documents if d["source"] == source
                            and d["publication_date"].startswith(year)
                            and d["candidate_relevant"] == candidate
                            and d["near_family_id"] not in used
                            and all("2019-01-01" <= day < "2023-01-01"
                                    for day in families[d["near_family_id"]]["publication_dates"])]
                    if not pool:
                        raise ValueError(f"Empty development stratum: {year}, {source}, {candidate}")
                    chosen = min(pool, key=lambda d: (
                        hashlib.sha256(("expand_v1:" + d["exact_group_id"]).encode()).hexdigest(),
                        d["document_id"]))
                    used.add(chosen["near_family_id"])
                    queue.append({k: chosen[k] for k in (
                        "source", "document_id", "record_version_hash", "exact_group_id", "near_family_id",
                        "publication_date", "title", "candidate_relevant")})
        selection = "expand_v1: one match and one nonmatch per FT/Guardian source/year in 2020–2022; deterministic SHA-256 order of exact groups; existing annotation/packet families excluded. Not representative evaluation."
    else:
        raise ValueError("Unknown mode")
    selected, annotations = [], []
    used = set()
    for number, q in enumerate(queue, 1):
        d = by_key[q["source"], q["document_id"]]
        if d["record_version_hash"] != q["record_version_hash"]:
            raise ValueError("Queue version changed; do not silently replace reviewed input.")
        if d["near_family_id"] in excluded:
            raise ValueError("Family already reserved for annotation or protected from development")
        if d["near_family_id"] in used:
            raise ValueError("Repeated family in packet")
        if not all("2019-01-01" <= day < "2023-01-01"
                   for day in families[d["near_family_id"]]["publication_dates"]):
            raise ValueError("Family contains material outside the development period")
        used.add(d["near_family_id"])
        selected.append(d)
        annotations.append({**q, "article_number": number, "batch_id": batch_id,
                            "body_hash": d["body_hash"], "status": "unreviewed",
                            "relevant": None, "threat": None, "disruption": None,
                            "deescalation": None, "location": None, "temporal_focus": None,
                            "annotator": None, "annotator_type": None, "human_reviewed": False,
                            "annotated_at": None, "guide_version": GUIDE_VERSION, "input_scope": "full_article",
                            "rationale": "", "evidence_excerpts": [], "evidence_spans": []})
    packet = [f"# {batch_id} — private development reading packet\n",
              "Licensed local text. Read without classifier predictions or market outcomes.\n"]
    for number, d in enumerate(selected, 1):
        packet += [f"## Article {number}: {d['title']}\n",
                   f"{d['source']} · {d['publication_date']} · {d['document_id']} · {d['record_version_hash']}\n",
                   d["body_clean"] + "\n"]
    manifest = {"batch_id": batch_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "purpose": "development_only", "mode": mode, "selection": selection,
                "excluded_family_ids": sorted(excluded),
                "input_sha256": {str(p.relative_to(root)): digest(p) for p in inputs},
                "source_counts": dict(Counter(d["source"] for d in selected)),
                "year_counts": dict(Counter(d["publication_date"][:4] for d in selected)),
                "timing": "Retrospective publication dates; historical text versions unverified; FT 30-day archive delay remains.",
                "coverage": "Known FT gaps accepted; no imputation or further collection prerequisite."}
    batch.mkdir(parents=True, exist_ok=False)
    write_rows(batch / "queue_snapshot.jsonl", queue)
    write_rows(batch / "reviewed_texts.jsonl", selected)
    write_rows(batch / "family_snapshot.jsonl", [families[d["near_family_id"]] for d in selected])
    write_rows(batch / "annotations.jsonl", annotations)
    (batch / "reading_packet.md").write_text("\n".join(packet), encoding="utf-8")
    manifest["frozen_artifact_sha256"] = {name: digest(batch / name) for name in (
        "queue_snapshot.jsonl", "reviewed_texts.jsonl", "family_snapshot.jsonl", "reading_packet.md")}
    (batch / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--mode", choices=("queue", "expand"), default="queue")
    args = parser.parse_args()
    print(prepare(args.batch_id, args.mode))
