"""Reserve a chronological NLP holdout without assigning reference labels.

Duplicate families and conservative report links form transitive components.
No relevance-screening result, reference label or market series selects the set.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

try:
    from .prepare_news_review import ROOT, digest, read, write_rows
except ImportError:
    from prepare_news_review import ROOT, digest, read, write_rows

RESERVATION = "nlp_eval_2023_v1"
FIELDS = ("source", "document_id", "record_version_hash", "body_hash", "exact_group_id",
          "near_family_id", "publication_date", "title")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def report_links(documents, config, historical_groups=()):
    """Conservative links for partition protection only, never label propagation."""
    links = []
    families = {d["near_family_id"] for d in documents}

    def connect(ids, reason, detail=None):
        ids = sorted(set(ids))
        if not set(ids) <= families:
            raise ValueError("Link refers to a family absent from the corpus")
        for other in ids[1:]:
            links.append(dict(left=ids[0], right=other, reason=reason, detail=detail))

    for reason, members in historical_groups:
        connect(members, reason)
    by_body = defaultdict(list)
    for d in documents:
        by_body[d["body_hash"]].append(d["near_family_id"])
    for ids in by_body.values():
        connect(ids, "identical_body_across_sources")
    for rule in config["related_report_rules"]:
        members = []
        for d in documents:
            if not rule["start"] <= d["publication_date"] < rule["end"]:
                continue
            body = d["body_clean"]
            if rule.get("lead_words"):
                body = " ".join(body.split()[:rule["lead_words"]])
            text = d["title"] + "\n" + body
            if re.search(rule["pattern"], text, re.I):
                members.append(d["near_family_id"])
        connect(members, "related_report_rule:" + rule["id"])
    # Existing detection is within source. Add conservative cross-source and
    # partial-copy links, including short summaries, in a fixed 14-day window.
    reps = {}
    for d in documents:
        reps.setdefault(d["body_hash"], d)
    items = sorted(reps.values(), key=lambda d: (d["publication_date"], d["source"], d["document_id"]))
    shingles = []
    for d in items:
        words = re.findall(r"\b\w+\b", d["body_clean"].lower())
        shingles.append({tuple(words[i:i+5]) for i in range(len(words)-4)})
    for i, a in enumerate(items):
        for j in range(i+1, len(items)):
            b = items[j]
            gap = (date.fromisoformat(b["publication_date"]) - date.fromisoformat(a["publication_date"])).days
            if gap > config["similarity_days"]:
                break
            if a["near_family_id"] == b["near_family_id"]:
                continue
            left, right = shingles[i], shingles[j]
            minimum = min(len(left), len(right))
            if minimum < config["minimum_shingles"]:
                continue
            shared = len(left & right)
            containment = shared / minimum
            if containment >= config["containment_threshold"]:
                connect([a["near_family_id"], b["near_family_id"]], "five_word_shingle_containment",
                        {"containment": round(containment, 6), "shared_shingles": shared, "days": gap})
    return sorted(links, key=lambda r: (r["left"], r["right"], r["reason"]))


def assign(documents, families, links, config, exposed_families=()):
    """Apply one partition to each connected component, then select its queue."""
    parent = {f["near_family_id"]: f["near_family_id"] for f in families}

    def find(x):
        if x not in parent:
            raise ValueError("Unknown family")
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in links:
        a, b = find(edge["left"]), find(edge["right"])
        parent[max(a, b)] = min(a, b)
    members = defaultdict(list)
    dates = defaultdict(set)
    for f in families:
        dates[find(f["near_family_id"])].update(f["publication_dates"])
    for d in documents:
        members[find(d["near_family_id"])].append(d)
        dates[find(d["near_family_id"])].add(d["publication_date"])
    exposed = {find(f) for f in exposed_families}
    start, end = config["evaluation_start"], config["evaluation_end_exclusive"]
    components, assignments = [], []
    for root, docs in sorted(members.items()):
        days = sorted(dates[root])
        ids = sorted({d["near_family_id"] for d in docs})
        group_id = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if days[0] < start or root in exposed:
            role = "development_linked"
        elif any(start <= d["publication_date"] < end and d["source"] in config["evaluation_sources"] for d in docs):
            role = "evaluation_reserved"
        else:
            role = "future_or_auxiliary_reserved"
        components.append(dict(split_group_id=group_id, partition=role, near_family_ids=ids,
                               document_count=len(docs), min_date=days[0], max_date=days[-1],
                               includes_prior_annotation_packet=root in exposed))
        for d in docs:
            candidate = start <= d["publication_date"] < end and d["source"] in config["evaluation_sources"]
            assignments.append({**{k: d[k] for k in FIELDS}, "split_group_id": group_id,
                "partition": role,
                "nlp_training_eligible": role == "development_linked" and config["development_start"] <= d["publication_date"] < start,
                "evaluation_candidate": candidate,
                "evaluation_eligible": candidate and role == "evaluation_reserved",
                "evaluation_selected": False,
                "exclusion_reason": "pre_cutoff_or_previously_reviewed_related_component" if candidate and role == "development_linked" else None})
    assignments.sort(key=lambda d: (d["publication_date"], d["source"], d["document_id"]))
    chosen = set()
    for a in assignments:
        if a["evaluation_eligible"] and a["near_family_id"] not in chosen:
            a["evaluation_selected"] = True
            chosen.add(a["near_family_id"])
    queue = [{**a, "article_number": i, "status": "reserved_unlabelled", "guide_version": "0.2"}
             for i, a in enumerate((a for a in assignments if a["evaluation_selected"]), 1)]
    return assignments, components, queue


def build(root, batch):
    config = json.loads((batch / "selection_config.json").read_text())
    docs_path = root / "data/processed/news_documents.jsonl"
    families_path = root / "data/processed/news_review_families.jsonl"
    docs, families = read(docs_path), read(families_path)
    inputs = [docs_path, families_path]
    exposed = set()
    for path in sorted((root / "data/annotations").glob("*/*.jsonl")):
        if path.name not in ("annotations.jsonl", "queue_snapshot.jsonl", "duplicate_review_texts.jsonl"):
            continue
        inputs.append(path)
        for row in read(path):
            matches = [d for d in docs if d["document_id"] == row["document_id"]
                       and (not row.get("source") or d["source"] == row["source"])]
            if len(matches) != 1 or row["record_version_hash"] not in matches[0]["version_hashes"]:
                raise ValueError("Prior exposure cannot be linked to the frozen corpus")
            exposed.add(matches[0]["near_family_id"])
    historical = []
    path = root / config["prior_episode_review"]
    inputs.append(path)
    for group in json.loads(path.read_text())["groups"]:
        historical.append(("prior_overlap:" + group["episode_tag"], [m["near_family_id"] for m in group["members"]]))
    path = root / config["prior_summary_review"]
    inputs.append(path)
    for link in json.loads(path.read_text())["links"]:
        historical.append(("prior_candidate_summary_link", [link["summary"]["near_family_id"], link["possible_full_story"]["near_family_id"]]))
    links = report_links(docs, config, historical)
    assignments, components, queue = assign(docs, families, links, config, exposed)
    return docs, families, inputs, links, assignments, components, queue


def reserve(root=ROOT, reservation_id=RESERVATION):
    batch = root / "data/evaluation" / reservation_id
    if (batch / "manifest.json").exists():
        raise ValueError("Existing reservation left unchanged")
    freeze = json.loads((batch / "guide_freeze.json").read_text())
    if digest(batch / freeze["guide_snapshot"]) != freeze["guide_sha256"]:
        raise ValueError("Frozen guide changed")
    docs, families, inputs, links, assignments, components, queue = build(root, batch)
    if not queue:
        raise ValueError("No eligible evaluation articles")
    write_rows(batch / "group_links.jsonl", links)
    write_rows(batch / "assignments.jsonl", assignments)
    write_rows(batch / "group_components.jsonl", components)
    write_rows(batch / "queue_snapshot.jsonl", queue)
    by_key = {(d["source"], d["document_id"]): d for d in docs}
    write_rows(batch / "evaluation_texts.jsonl", [by_key[q["source"], q["document_id"]] for q in queue])
    reserved = {a["near_family_id"] for a in assignments if a["partition"] == "evaluation_reserved"}
    write_rows(batch / "family_snapshot.jsonl", [f for f in families if f["near_family_id"] in reserved])
    manifest = dict(reservation_id=reservation_id, status="reserved_unlabelled",
                    created_at=datetime.now(timezone.utc).isoformat(), guide_version="0.2",
                    guide_sha256=freeze["guide_sha256"], guide_frozen_at=freeze["guide_frozen_at"],
                    evaluation_articles=len(queue), evaluation_families=len(queue),
                    evaluation_related_components=len({q["split_group_id"] for q in queue}),
                    protected_documents=sum(a["partition"] == "evaluation_reserved" for a in assignments),
                    source_counts=dict(Counter(q["source"] for q in queue)),
                    quarter_counts=dict(Counter(str((int(q["publication_date"][5:7])-1)//3+1) for q in queue)),
                    candidate_documents=sum(a["evaluation_candidate"] for a in assignments),
                    excluded_candidate_documents=sum(a["evaluation_candidate"] and not a["evaluation_eligible"] for a in assignments),
                    input_sha256={str(p.relative_to(root)): digest(p) for p in inputs},
                    frozen_artifact_sha256={p.name: digest(p) for p in sorted(batch.iterdir()) if p.is_file()},
                    code_sha256=digest(Path(__file__)))
    save_json(batch / "manifest.json", manifest)
    return manifest


def validate(root=ROOT, reservation_id=RESERVATION):
    batch = root / "data/evaluation" / reservation_id
    manifest = json.loads((batch / "manifest.json").read_text())
    for name, expected in manifest["frozen_artifact_sha256"].items():
        if digest(batch / name) != expected:
            raise ValueError(f"Frozen evaluation artifact changed: {name}")
    for name, expected in manifest["input_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Evaluation input changed; explicit re-audit required: {name}")
    preservation = batch / "preservation_before.json"
    if preservation.exists():
        for name, expected in json.loads(preservation.read_text()).items():
            if digest(root / name) != expected:
                raise ValueError(f"Previously saved data changed: {name}")
    freeze = json.loads((batch / "guide_freeze.json").read_text())
    if not (freeze["guide_sha256"] == manifest["guide_sha256"] == digest(batch / freeze["guide_snapshot"])):
        raise ValueError("Guide freeze mismatch")
    if datetime.fromisoformat(freeze["guide_frozen_at"]) > datetime.fromisoformat(manifest["created_at"]):
        raise ValueError("Guide was not frozen before reservation")
    docs, families, inputs, links, assignments, components, queue = build(root, batch)
    for name, expected in (("group_links.jsonl", links), ("assignments.jsonl", assignments),
                           ("group_components.jsonl", components), ("queue_snapshot.jsonl", queue)):
        if read(batch / name) != expected:
            raise ValueError(f"Reservation cannot be reproduced: {name}")
    selected = {(q["source"], q["document_id"]): q for q in queue}
    texts = read(batch / "evaluation_texts.jsonl")
    current = {(d["source"], d["document_id"]): d for d in docs}
    if len(texts) != len(selected) or any(d != current.get((d["source"], d["document_id"])) for d in texts):
        raise ValueError("Frozen evaluation texts do not match corpus")
    if {(d["source"], d["document_id"]) for d in texts} != set(selected):
        raise ValueError("Evaluation queue/text identity mismatch")
    by_family = {a["near_family_id"]: a for a in assignments}
    for link in links:
        if by_family[link["left"]]["split_group_id"] != by_family[link["right"]]["split_group_id"]:
            raise ValueError("Related reports split across groups")
    if any(q["nlp_training_eligible"] or q["status"] != "reserved_unlabelled" for q in queue):
        raise ValueError("Evaluation contamination")
    expected_counts = dict(evaluation_articles=len(queue), evaluation_families=len(queue),
                           evaluation_related_components=len({q["split_group_id"] for q in queue}),
                           protected_documents=sum(a["partition"] == "evaluation_reserved" for a in assignments),
                           source_counts=dict(Counter(q["source"] for q in queue)),
                           quarter_counts=dict(Counter(str((int(q["publication_date"][5:7])-1)//3+1) for q in queue)),
                           candidate_documents=sum(a["evaluation_candidate"] for a in assignments),
                           excluded_candidate_documents=sum(a["evaluation_candidate"] and not a["evaluation_eligible"] for a in assignments))
    if any(manifest[k] != v for k, v in expected_counts.items()):
        raise ValueError("Reservation summary does not match assignments")
    return {k: manifest[k] for k in ("reservation_id", "status", "evaluation_articles", "evaluation_families",
            "evaluation_related_components", "protected_documents", "source_counts", "quarter_counts",
            "candidate_documents", "excluded_candidate_documents", "guide_sha256")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate() if args.validate else reserve(), indent=2))
