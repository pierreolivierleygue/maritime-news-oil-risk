"""Retrieve and validate the complete 2019–2025 Guardian query corpus.

Completed months are cached separately from the original five-article audit.
Reruns resume missing months; --offline validates and rebuilds without API access.
"""

import argparse
import calendar
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import unicodedata

from audit_data import ROOT, QUERY, fetch, guardian_key, stamp

ENDPOINT = "https://content.guardianapis.com/search"
CACHE = ROOT / "data/raw/guardian_full"
OUTPUT = ROOT / "data/processed/guardian_articles.jsonl"
REPORT = ROOT / "data/audit/guardian_retrieval.json"
PAGE_SIZE = 100


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def params_for(month):
    year, number = map(int, month.split("-"))
    return {
        "q": QUERY, "from-date": month + "-01",
        "to-date": f"{month}-{calendar.monthrange(year, number)[1]}",
        "page-size": PAGE_SIZE, "order-by": "oldest", "type": "article",
        "show-fields": "bodyText,lastModified",
    }


def validate_month(payload, month):
    """Reject partial, drifting, repeated or out-of-window page collections."""
    if payload.get("endpoint") != ENDPOINT or payload.get("params") != params_for(month):
        raise RuntimeError(f"{month}: cached query/settings differ; preserve and review cache.")
    pages = payload["pages"]
    if not pages:
        raise RuntimeError(f"{month}: no response pages.")
    total = pages[0]["response"]["total"]
    if type(total) is not int or total < 0:
        raise RuntimeError(f"{month}: invalid total.")
    count = (total + PAGE_SIZE - 1) // PAGE_SIZE
    if len(pages) != max(1, count):
        raise RuntimeError(f"{month}: incomplete pagination.")
    records = []
    for number, page in enumerate(pages, 1):
        response = page["response"]
        retrieved = datetime.fromisoformat(page["retrieved_at"].replace("Z", "+00:00"))
        if retrieved.tzinfo is None:
            raise RuntimeError(f"{month}: missing retrieval timezone.")
        if (response.get("status") != "ok" or response.get("total") != total
                or response.get("pages") != count or response.get("currentPage") != number
                or response.get("pageSize") != PAGE_SIZE or response.get("orderBy") != "oldest"):
            raise RuntimeError(f"{month}: inconsistent page metadata or changing totals.")
        expected = min(PAGE_SIZE, max(0, total - (number - 1) * PAGE_SIZE))
        if len(response["results"]) != expected:
            raise RuntimeError(f"{month}: truncated page.")
        for article in response["results"]:
            published = article.get("webPublicationDate", "")
            parsed = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if (not article.get("id") or not article.get("webTitle")
                    or article.get("type") != "article" or not published.startswith(month + "-")
                    or parsed.tzinfo is None):
                raise RuntimeError(f"{month}: invalid article identity, type or publication date.")
            records.append((article, page["retrieved_at"], number))
    if len({r[0]["id"] for r in records}) != total:
        raise RuntimeError(f"{month}: repeated IDs or incomplete unique count.")
    return records


class Client:
    def __init__(self):
        self.key = None
        self.last_request = 0.0
        self.calls = 0

    def __call__(self, params):
        if self.key is None:
            self.key = guardian_key()
        for attempt in range(3):
            if self.calls >= 450:
                raise RuntimeError("Run request budget reached; completed months are cached.")
            time.sleep(max(0, 1.1 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            self.calls += 1
            try:
                return json.loads(fetch(ENDPOINT, {**params, "api-key": self.key}))["response"]
            except RuntimeError as exc:
                message = str(exc)  # fetch() deliberately removes authenticated URLs.
                transient = any(s in message for s in (
                    "timed out", "connection failed", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504"))
                if not transient or attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))


def collect_month(month, cache, client, offline=False):
    path = cache / f"{month}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload, validate_month(payload, month)
    if offline:
        raise RuntimeError(f"{month}: missing complete cache; offline rebuild stopped.")
    params = params_for(month)
    pages = []
    number = 1
    while True:
        response = client({**params, "page": number})
        if response.get("status") != "ok":
            raise RuntimeError(f"{month}: unsuccessful API response.")
        pages.append({"retrieved_at": stamp(), "response": response})
        first = pages[0]["response"]
        if response.get("total") != first.get("total") or response.get("pages") != first.get("pages"):
            raise RuntimeError(f"{month}: totals changed during pagination; month not cached.")
        if number >= max(1, first["pages"]):
            break
        number += 1
    payload = {"endpoint": ENDPOINT, "params": params, "pages": pages}
    records = validate_month(payload, month)
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload, records


def run(offline=False):
    months = [f"{year}-{month:02d}" for year in range(2019, 2026) for month in range(1, 13)]
    prior_path = ROOT / "data/audit/guardian_summary.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    old_counts = {p["month"]: p["total_matches"] for p in prior.get("periods", [])} if prior.get("query") == QUERY else {}
    summary = {
        "started_at": stamp(), "status": "incomplete", "endpoint": ENDPOINT,
        "query_version": "guardian_q1", "query": QUERY, "requested_months": months,
        "periods": [], "pending_months": months.copy(),
        "limitations": [
            "Complete means all API matches for this query, type and month at retrieval, not all relevant reporting.",
            "Monthly caches may have different retrieval dates; they are not a simultaneous archive snapshot.",
            "Current text and lastModified do not reconstruct original historical versions or availability.",
            "No relevance labels or weekly financial predictors have been inferred.",
            "Near-duplicate detection is not implemented.",
        ],
    }
    client = Client()
    articles = []
    ids = set()
    def checkpoint():
        summary["updated_at"] = stamp()
        summary["network_requests_this_run"] = client.calls
        atomic_write(REPORT, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    checkpoint()
    for month in months:
        payload, records = collect_month(month, CACHE, client, offline)
        path = CACHE / f"{month}.json"
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for raw, retrieved, page in records:
            if raw["id"] in ids:
                raise RuntimeError(f"{month}: duplicate ID across monthly partitions.")
            ids.add(raw["id"])
            fields = raw.get("fields", {})
            body = fields.get("bodyText") or ""
            clean = " ".join(unicodedata.normalize("NFC", body).split())
            modified = fields.get("lastModified")
            flags = ["historical_text_version_unverified"]
            if not clean:
                flags.append("body_missing")
            if modified:
                delta = datetime.fromisoformat(modified.replace("Z", "+00:00")) - datetime.fromisoformat(raw["webPublicationDate"].replace("Z", "+00:00"))
                if delta.total_seconds() > 86400:
                    flags.append("updated_more_than_one_day_after_publication")
            else:
                flags.append("last_modified_missing")
            articles.append({
                "document_id": raw["id"], "record_version_hash": sha(encode(raw)),
                "source": "Guardian Content API", "title": raw["webTitle"],
                "web_url": raw.get("webUrl"), "section_id": raw.get("sectionId"),
                "publication_datetime": raw["webPublicationDate"],
                "publication_date": raw["webPublicationDate"][:10],
                "record_updated_at": modified, "retrieved_at": retrieved,
                "query_version": "guardian_q1", "body_raw": body, "body_clean": clean,
                "body_hash": sha(clean), "word_count": len(clean.split()), "quality_flags": flags,
                "provenance": {"cache_file": str(path.relative_to(ROOT)), "cache_sha256": file_hash,
                               "page": page, "month": month},
            })
        total = len(records)
        summary["periods"].append({
            "month": month, "status": "complete" if total else "confirmed_zero_results",
            "total_matches": total, "unique_ids": total, "pages": len(payload["pages"]),
            "articles_with_body": sum(bool(r[0].get("fields", {}).get("bodyText", "").strip()) for r in records),
            "prior_audit_total": old_counts.get(month),
            "change_since_prior_audit": total - old_counts[month] if month in old_counts else None,
            "retrieved_at": [p["retrieved_at"] for p in payload["pages"]],
            "cache_file": str(path.relative_to(ROOT)), "cache_sha256": file_hash,
        })
        summary["pending_months"].remove(month)
        checkpoint()
        print(f"{month}: {total} unique articles; reconciled.", flush=True)
    articles.sort(key=lambda r: (r["publication_datetime"], r["document_id"]))
    bodies = defaultdict(list)
    for article in articles:
        if article["body_clean"]:
            bodies[article["body_hash"]].append(article["document_id"])
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in articles)
    atomic_write(OUTPUT, text)
    summary.update(
        status="complete", completed_at=stamp(), article_count=len(articles), unique_ids=len(ids),
        articles_with_body=sum(bool(r["body_clean"]) for r in articles),
        year_counts=dict(sorted(Counter(r["publication_date"][:4] for r in articles).items())),
        quality_flags=dict(Counter(f for r in articles for f in r["quality_flags"])),
        exact_body_groups={h: values for h, values in bodies.items() if len(values) > 1},
        zero_match_months=[p["month"] for p in summary["periods"] if not p["total_matches"]],
        output=str(OUTPUT.relative_to(ROOT)), output_sha256=sha(text),
    )
    checkpoint()
    print(f"Complete: {len(articles)} unique articles; {summary['articles_with_body']} bodies. Saved {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require all complete cached months; make no network requests.")
    args = parser.parse_args()
    try:
        run(args.offline)
    except RuntimeError as exc:
        parser.exit(1, f"Retrieval stopped: {exc}\nCompleted months retained; rerun to resume.\n")
    except (ValueError, KeyError, TypeError, OSError):
        parser.exit(1, "Retrieval stopped: invalid response, cache or storage error. No authenticated URLs printed.\n")


if __name__ == "__main__":
    main()
