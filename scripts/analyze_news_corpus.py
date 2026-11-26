"""Group duplicate news, apply provisional relevance rules, and audit weekly coverage.

No network, model fitting, market outcomes or manual annotations are used. Known
collection gaps are retained as flags, not blockers. Counts are retrospective.
"""

from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import itertools
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
START, END = date(2019, 1, 1), date(2025, 12, 31)
FT = "Financial Times; London (UK)"
GUARDIAN = "The Guardian"
RULE_VERSION = "maritime_candidate_v0.1"
RULES = {
    "geography": r"\bhormuz\b|\bred\s+sea\b|\bbab[\s-]+(?:el|al)[\s-]+mand(?:eb|ab)\b",
    "transport": r"\b(?:ship\w*|tanker\w*|vessel\w*|maritime|cargo|route\w*|strait\w*|canal\w*|port\w*|passage|pipeline\w*)\b",
    "risk_or_response": r"\b(?:attack\w*|seiz\w*|detain\w*|hijack\w*|pirac\w*|pirate\w*|threat\w*|disrupt\w*|blockad\w*|block\w*|clos\w*|shut\w*|rerout\w*|divert\w*|suspend\w*|missile\w*|drone\w*|houthi\w*|security|escort\w*|resum\w*|reopen\w*|halt\w*|risk\w*)\b",
}
WINDOW_CHARS = 700
NEAR_THRESHOLD, NEAR_DAYS = 0.80, 14
PATTERNS = {k: re.compile(v, re.I) for k, v in RULES.items()}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_jsonl(name, rows):
    save(ROOT / "data/processed" / name,
         "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def screen(title, body):
    """A co-occurrence screen, never a semantic threat/disruption decision."""
    text = " ".join((title + " " + body).split())
    matches = {name: list(pattern.finditer(text)) for name, pattern in PATTERNS.items()}
    for geo in matches["geography"]:
        nearby = {k: [m for m in matches[k] if abs(m.start() - geo.start()) <= WINDOW_CHARS]
                  for k in ("transport", "risk_or_response")}
        for transport, risk in itertools.product(nearby["transport"], nearby["risk_or_response"]):
            first = min(geo.start(), transport.start(), risk.start())
            last = max(geo.end(), transport.end(), risk.end())
            if last - first <= WINDOW_CHARS:
                return {"rule_version": RULE_VERSION, "candidate_relevant": True,
                        "status": "rule_match_requires_review", "relevant": None,
                        "threat": None, "disruption": None,
                        "matched_terms": {"geography": geo.group(), "transport": transport.group(),
                                          "risk_or_response": risk.group()},
                        "evidence_window": text[max(0, first - 100):min(len(text), last + 100)]}
    return {"rule_version": RULE_VERSION, "candidate_relevant": False,
            "status": "no_rule_match_not_verified_irrelevant", "relevant": None,
            "threat": None, "disruption": None, "matched_terms": {}, "evidence_window": None}


def shingle_set(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 50:
        return set()
    return {tuple(words[n:n + 5]) for n in range(len(words) - 4)}


def jaccard(left, right):
    return len(left & right) / len(left | right) if left and right else 0.0


def monday(day):
    return day - timedelta(days=day.weekday())


def split_name(day):
    return "development" if day < "2023-01-01" else "validation" if day < "2024-01-01" else "test"


def combine_versions(records):
    by_id = defaultdict(list)
    for record in records:
        by_id[(record["source"], record["document_id"])].append(record)
    documents = []
    for (source, doc), variants in sorted(by_id.items()):
        # Metadata-only variants can share one analysis row without losing provenance.
        if len({(r["body_hash"], r["publication_date"]) for r in variants}) != 1:
            raise ValueError("Same-ID body/date conflict requires an explicit version decision.")
        representative = min(variants, key=lambda r: r["record_version_hash"])
        documents.append({**representative,
                          "document_key": digest(source + "\n" + doc),
                          "version_hashes": sorted({r["record_version_hash"] for r in variants}),
                          "title_variants": sorted({r["title"] for r in variants}),
                          "version_selection": "identical_body_and_date; lexicographic_hash_metadata_representative"})
    return documents


def collection_status(source, first, last, ft_status, guardian_months):
    if source == GUARDIAN:
        months = set()
        day = first
        while day <= last:
            months.add(day.isoformat()[:7])
            day += timedelta(days=1)
        return "query_collection_complete" if months <= guardian_months else "collection_unverified"
    if source == FT:
        statuses = {ft_status.get(y, "unverified") for y in range(first.year, last.year + 1)}
        return "annual_counts_reconciled_filters_unverified" if statuses == {"counts_reconciled"} else "known_annual_gap_accepted"
    return "supplementary_source_timing_unverified"


def weekly_rows(documents, sources, ft_status, guardian_months):
    buckets = defaultdict(list)
    for doc in documents:
        day = date.fromisoformat(doc["publication_date"])
        if START <= day <= END:
            buckets[(doc["source"], monday(day))].append(doc)
    output = []
    week = monday(START)
    while week <= END:
        first, last = max(week, START), min(week + timedelta(days=6), END)
        for source in sources:
            docs = buckets[(source, week)]
            groups = {d["exact_group_id"] for d in docs}
            candidates = {d["exact_group_id"] for d in docs if d["candidate_relevant"]}
            status = collection_status(source, first, last, ft_status, guardian_months)
            output.append({
                "source": source, "week_start": week.isoformat(),
                "week_end": (week + timedelta(days=6)).isoformat(),
                "observed_start": first.isoformat(), "observed_end": last.isoformat(),
                "partial_boundary_week": first != week or last != week + timedelta(days=6),
                "observed_document_count": len(docs), "exact_text_count": len(groups),
                "within_week_exact_copies_removed": len(docs) - len(groups),
                "candidate_exact_text_count": len(candidates),
                "candidate_share": len(candidates) / len(groups) if groups else None,
                "near_family_count_sensitivity": len({d["near_family_id"] for d in docs}),
                "candidate_near_family_count_sensitivity": len({d["near_family_id"] for d in docs if d["candidate_relevant"]}),
                "collection_status": status,
                "zero_interpretation": ("not_zero" if docs else "confirmed_no_query_matches" if status == "query_collection_complete"
                                        else "no_observed_records_completeness_not_verified"),
                "timing_basis": "retrospective_publication_date_not_archive_availability",
                "weekly_model_eligible_source": source in (FT, GUARDIAN),
                "exact_group_ids": sorted(groups),
            })
        week += timedelta(days=7)
    return output


def run():
    paths = [ROOT / "data/processed/proquest_articles.jsonl", ROOT / "data/processed/guardian_articles.jsonl"]
    records = []
    for path in paths:
        for r in read_jsonl(path):
            source = GUARDIAN if path.name.startswith("guardian") else r["publication_group"] or r["publication_raw"]
            if not r["publication_date"] or not r["body_clean"]:
                raise ValueError("Missing date/body requires a documented exclusion before analysis.")
            records.append({**r, "source": source, "input_file": str(path.relative_to(ROOT))})
    documents = combine_versions(records)
    groups = defaultdict(list)
    for doc in documents:
        group_id = digest(doc["source"] + "\n" + doc["body_hash"])
        doc["exact_group_id"] = group_id
        groups[group_id].append(doc)
    summaries, texts = {}, {}
    for group_id, members in sorted(groups.items()):
        # Union title variants for the screen so an arbitrary edition headline cannot
        # determine the label of an otherwise identical full text.
        titles = sorted({t for m in members for t in m["title_variants"]})
        results = [screen(title, members[0]["body_clean"]) for title in titles]
        screening = next((r for r in results if r["candidate_relevant"]), results[0])
        dates = sorted({m["publication_date"] for m in members})
        summary = {"exact_group_id": group_id, "source": members[0]["source"],
                   "document_ids": sorted(m["document_id"] for m in members),
                   "version_count": sum(len(m["version_hashes"]) for m in members),
                   "body_hash": members[0]["body_hash"], "publication_dates": dates,
                   "title_variants": titles, "word_count": members[0]["word_count"],
                   "short_text_under_100_words": members[0]["word_count"] < 100,
                   "screening": screening}
        summaries[group_id] = summary
        texts[group_id] = shingle_set(members[0]["body_clean"])
        for member in members:
            member["candidate_relevant"] = screening["candidate_relevant"]
    parent = {group_id: group_id for group_id in groups}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    edges = []
    for source in sorted({d["source"] for d in documents}):
        group_ids = sorted(g for g in summaries if summaries[g]["source"] == source)
        for a, b in itertools.combinations(group_ids, 2):
            left, right = texts[a], texts[b]
            if not left or not right or min(len(left), len(right)) / max(len(left), len(right)) < NEAR_THRESHOLD:
                continue
            if min(abs((date.fromisoformat(x) - date.fromisoformat(y)).days)
                   for x in summaries[a]["publication_dates"] for y in summaries[b]["publication_dates"]) > NEAR_DAYS:
                continue
            similarity = jaccard(left, right)
            if similarity >= NEAR_THRESHOLD:
                edges.append({"left": a, "right": b, "source": source, "jaccard": round(similarity, 6),
                              "status": "unreviewed_near_duplicate_candidate"})
                ra, rb = find(a), find(b)
                parent[max(ra, rb)] = min(ra, rb)
    families = defaultdict(list)
    for g in groups:
        families[find(g)].append(g)
    family_rows = []
    for root, members in sorted(families.items()):
        dates = sorted({day for g in members for day in summaries[g]["publication_dates"]})
        family_rows.append({"near_family_id": root, "source": summaries[root]["source"],
                            "exact_group_ids": sorted(members), "publication_dates": dates,
                            "crosses_provisional_split": len({split_name(day) for day in dates}) > 1,
                            "status": "unreviewed_near_duplicate_component" if len(members) > 1 else "exact_only"})
        for group_id in members:
            summaries[group_id]["near_family_id"] = root
            for doc in groups[group_id]:
                doc["near_family_id"] = root
    ft_audit = json.loads((ROOT / "data/audit/proquest_ft_collection.json").read_text())
    guardian_audit = json.loads((ROOT / "data/audit/guardian_retrieval.json").read_text())
    guardian_months = {p["month"] for p in guardian_audit["periods"] if p["status"] in ("complete", "confirmed_zero_results")}
    ft_status = {p["year"]: p["status"] for p in ft_audit["annual"]}
    sources = sorted({d["source"] for d in documents})
    weekly = weekly_rows(documents, sources, ft_status, guardian_months)
    annual = []
    for source in sources:
        for year in range(2019, 2026):
            docs = [d for d in documents if d["source"] == source and d["publication_date"].startswith(str(year))]
            weeks = [w for w in weekly if w["source"] == source and w["observed_end"].startswith(str(year))]
            annual.append({"source": source, "year": year, "document_count": len(docs),
                           "exact_text_count": len({d["exact_group_id"] for d in docs}),
                           "candidate_exact_text_count": len({d["exact_group_id"] for d in docs if d["candidate_relevant"]}),
                           "weeks": len(weeks), "weeks_with_observed_text": sum(w["exact_text_count"] > 0 for w in weeks),
                           "weeks_with_candidates": sum(w["candidate_exact_text_count"] > 0 for w in weeks)})
    metadata = {
        "input_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule_version": RULE_VERSION, "rules": RULES, "cooccurrence_window_characters": WINDOW_CHARS,
        "near_duplicate_rule": {"metric": "Jaccard over lowercased five-word shingles", "threshold": NEAR_THRESHOLD,
                                "minimum_words": 50, "maximum_date_gap_days": NEAR_DAYS, "within_source_only": True},
        "known_gaps_policy": "Proceed with observed records at user request; retain coverage flags; no imputation.",
        "raw_record_versions": len(records), "unique_source_document_ids": len(documents),
        "exact_source_text_groups": len(groups), "near_duplicate_edges": len(edges),
        "near_duplicate_components_with_multiple_texts": sum(len(f["exact_group_ids"]) > 1 for f in family_rows),
        "families_crossing_provisional_splits": sum(f["crosses_provisional_split"] for f in family_rows),
        "annual": annual,
        "limitations": [
            "Screening is unvalidated lexical co-occurrence, not reviewed relevance or event labels.",
            "Negation, historical background and incidental mentions can yield false positives; unmatched texts can be relevant.",
            "Threat and disruption are deliberately left null. Candidate counts are not the final risk index.",
            "Near-duplicate components are review/split-protection groups, not automatically approved story merges.",
            "Repeated independent reporting and different sources are retained; no combined-source risk share is computed.",
            "Exact groups are counted once per source/week, not permanently removed on later publication dates.",
            "Weekly FT counts include known incomplete years; missing matches are not inferred or filled.",
            "Only Guardian empty weeks with complete retrieval are confirmed zero query matches.",
            "Specialist sources remain supplementary because issue timing is unverified.",
            "Calendar-year article counts and week-year coverage use different boundaries; weeks use the clipped observed end date.",
            "The first and last weeks are partial. All timing is retrospective publication-date timing.",
        ],
    }
    save_jsonl("news_documents.jsonl", documents)
    save_jsonl("news_exact_groups.jsonl", list(summaries.values()))
    save_jsonl("news_near_duplicate_pairs.jsonl", edges)
    save_jsonl("news_review_families.jsonl", family_rows)
    save_jsonl("news_weekly_coverage.jsonl", weekly)
    # Make the review work concrete without treating rule output as annotations.
    review_lines = ["# Near-duplicate candidates\n",
                    "Unreviewed pairs within sources. Exact copies have already been grouped. "
                    "Connected families protect later splits; they are not approved story merges.\n",
                    "| Source | Dates | Similarity | Left title | Right title |\n|---|---|---:|---|---|"]
    for edge in sorted(edges, key=lambda e: (-e["jaccard"], e["left"], e["right"])):
        a, b = summaries[edge["left"]], summaries[edge["right"]]
        left_title, right_title = (" ".join(g["title_variants"][0].split()).replace("|", "\\|") for g in (a, b))
        review_lines.append(f"| {edge['source']} | {a['publication_dates'][0]} / {b['publication_dates'][0]} | {edge['jaccard']:.3f} | {left_title} | {right_title} |")
    save(ROOT / "data/audit/news_duplicate_review.md", "\n".join(review_lines) + "\n")
    early_families = {f["near_family_id"] for f in family_rows
                      if all(split_name(day) == "development" for day in f["publication_dates"])}
    queue, used_families = [], set()
    for source in (FT, GUARDIAN):
        for candidate in (True, False):
            pool = sorted((g for g in summaries.values() if g["source"] == source
                           and g["screening"]["candidate_relevant"] == candidate
                           and g["near_family_id"] in early_families),
                          key=lambda g: digest("review_v1:" + g["exact_group_id"]))
            selected = 0
            for group in pool:
                if group["near_family_id"] in used_families:
                    continue
                used_families.add(group["near_family_id"])
                member = min(groups[group["exact_group_id"]], key=lambda d: d["document_id"])
                queue.append({"review_number": len(queue) + 1, "source": source,
                              "document_id": member["document_id"], "record_version_hash": member["record_version_hash"],
                              "exact_group_id": group["exact_group_id"], "near_family_id": group["near_family_id"],
                              "publication_date": member["publication_date"], "title": member["title"],
                              "candidate_relevant": candidate, "status": "unreviewed",
                              "purpose": "development_rule_error_review_not_representative_evaluation"})
                selected += 1
                if selected == 3:
                    break
    save_jsonl("news_screening_review_queue.jsonl", queue)
    metadata["development_review_queue_size"] = len(queue)
    metadata["reference_audit_hashes"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "data/audit/proquest_ft_collection.json", ROOT / "data/audit/guardian_retrieval.json")}
    save(ROOT / "data/audit/news_corpus_analysis.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Corpus screening and weekly coverage\n",
             "Known collection gaps are accepted for continued work. Counts use observed records; gaps remain flagged.\n",
             "This is a provisional keyword screen, not reviewed relevance or a threat/disruption index.\n",
             f"{len(records)} record versions → {len(documents)} source/document IDs → {len(groups)} within-source exact text groups.\n",
             f"Near-duplicate review: {len(edges)} pairs, {metadata['near_duplicate_components_with_multiple_texts']} multi-text families. No automatic near-duplicate deletion.\n",
             "| Source | Year | Documents | Exact texts | Candidate texts | Weeks with candidates / observed weeks |\n|---|---:|---:|---:|---:|---:|"]
    for a in annual:
        if a["source"] in (FT, GUARDIAN):
            lines.append(f"| {a['source']} | {a['year']} | {a['document_count']} | {a['exact_text_count']} | {a['candidate_exact_text_count']} | {a['weeks_with_candidates']} / {a['weeks']} |")
    lines += ["\n## Interpretation and limits\n"] + ["- " + note for note in metadata["limitations"]]
    lines += ["\n## Reproduce\n", "```sh\n.venv/bin/python scripts/analyze_news_corpus.py\n```\n",
              "Detailed outputs are in data/processed/news_*.jsonl; configuration and hashes are in data/audit/news_corpus_analysis.json. Raw text and existing annotations remain unchanged.\n"]
    save(ROOT / "CORPUS_COVERAGE.md", "\n".join(lines))
    print(f"Processed {len(documents)} document IDs, {len(groups)} exact groups, {len(edges)} near-duplicate pairs and {len(weekly)} source-week rows.")


if __name__ == "__main__":
    run()
