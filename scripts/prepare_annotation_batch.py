"""Prepare a local ten-article development exercise without overwriting labels."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "data/annotations/proquest_dev_001"


def main():
    if BATCH.exists():
        raise SystemExit(f"Batch already exists; left unchanged: {BATCH}")
    corpus = ROOT / "data/processed/proquest_articles.jsonl"
    # JSONL records are newline-delimited; Unicode separators inside text are data.
    with corpus.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    early = sorted(
        [r for r in rows if "2019" <= (r["publication_date"] or "")[:4] <= "2022"],
        key=lambda r: (r["publication_date"], r["document_id"], r["record_version_hash"]),
    )
    # This known example is an explicit development choice, not a random draw.
    anchors = [r for r in early if r["document_id"] == "2297034993"]
    if len(anchors) != 1:
        raise SystemExit("Expected one version of the development anchor; review corpus.")
    if len({r["document_id"] for r in early}) != len(early):
        raise SystemExit("Multiple early-period versions need review before sampling.")
    selected = anchors.copy()
    rng = random.Random(42)
    strata = [
        ("2019", "Oil & Gas News; Manama", 1),
        ("2019", "Egypt Oil & Gas; Cairo", 1),
        ("2020", "Oil & Gas News; Manama", 3),
        ("2021", "Oil & Gas News; Manama", 2),
        ("2022", "Oil & Gas News; Manama", 2),
    ]
    for year, source, count in strata:
        pool = [r for r in early if r["publication_date"].startswith(year)
                and r["publication_raw"] == source and r not in selected]
        if len(pool) < count:
            raise SystemExit(f"Insufficient records for {year}, {source}.")
        selected.extend(rng.sample(pool, count))
    selected = selected[:1] + sorted(selected[1:], key=lambda r: (
        r["publication_date"], r["document_id"]))
    if len({r["body_hash"] for r in selected}) != 10:
        raise SystemExit("Repeated exact bodies in selection; review before proceeding.")

    annotations = []
    packet = [
        "# First annotation exercise — 10 articles\n",
        "Local licensed text: keep this packet private. Development only.\n",
        "Read each full article without checking OVX or subsequent events. Use "
        "ANNOTATION_GUIDE.md v0.2. Dates shown are archive fields, not verified availability.\n",
        "For each article provide relevant, threat, disruption (1=yes, 0=no, "
        "null=uncertain), plus a short rationale. If irrelevant, threat and disruption "
        "are both 0. An attack alone does not establish operational disruption.\n",
        "Reply in chat, for example: `Article 1: relevant=..., threat=..., "
        "disruption=...; rationale: ...`. We can then discuss ambiguity and save "
        "your reviewed labels in annotations.jsonl. All fields start unreviewed; "
        "null is not a negative label. Context fields can be completed during review.\n",
        "Selection: one previously inspected development example (2297034993), "
        "plus nine seeded draws across year/source strata. This is not an evaluation "
        "sample or an estimate of class prevalence. No model labels are supplied.\n",
    ]
    for number, r in enumerate(selected, 1):
        annotations.append({
            "batch_id": BATCH.name, "article_number": number,
            "document_id": r["document_id"], "record_version_hash": r["record_version_hash"],
            "status": "unreviewed", "relevant": None, "threat": None,
            "disruption": None, "location": None, "temporal_focus": None,
            "deescalation": None, "rationale": "", "annotator": None,
            "annotated_at": None, "guide_version": "0.2",
        })
        packet.extend([
            f"\n## Article {number}: {r['title'].replace(chr(10), ' ')}\n",
            f"Source: {r['publication_raw']} · Publication date: {r['publication_date']}  \n"
            f"ProQuest ID: {r['document_id']} · Words: {r['word_count']}  \n"
            f"Version: `{r['record_version_hash']}`\n",
            r["body_raw"] + "\n",
        ])
    manifest = {
        "batch_id": BATCH.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "development_only", "seed": 42,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "guide_sha256": hashlib.sha256((ROOT / "ANNOTATION_GUIDE.md").read_bytes()).hexdigest(),
        "early_pool_size": len(early), "anchor_document_id": "2297034993",
        "selection": "Known development anchor, then random draws without replacement in sorted year/source pools.",
        "random_draw_strata": [{"year": y, "source": s, "count": n} for y, s, n in strata],
        "year_counts": dict(Counter(r["publication_date"][:4] for r in selected)),
        "source_counts": dict(Counter(r["publication_raw"] for r in selected)),
        "records": [{k: r[k] for k in ("document_id", "record_version_hash", "body_hash")}
                    for r in selected],
        "limitations": ["Purposive development sample; not representative.",
                        "Near-duplicate detection not performed.",
                        "Original publication timing remains unverified."],
    }
    BATCH.mkdir(parents=True, exist_ok=False)
    for name, content in {
        "reading_packet.md": "\n".join(packet),
        "annotations.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in annotations),
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    }.items():
        (BATCH / name).write_text(content, encoding="utf-8")
    print(f"Prepared {len(selected)} unreviewed articles: {BATCH / 'reading_packet.md'}")


if __name__ == "__main__":
    main()
