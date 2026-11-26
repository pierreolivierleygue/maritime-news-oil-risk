"""Import French-labelled ProQuest TXT exports locally, without network access.

Rebuilds derived JSONL and an audit from unchanged raw exports. Preserves distinct
versions of an ID and groups exact bodies without treating them as new events.
"""

import argparse
import calendar
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
SEPARATOR = re.compile(r"^_{20,}[ \t]*$", re.M)
BODY_START = re.compile(r"^Texte intégral:[ \t]*", re.M)
METADATA_START = re.compile(
    r"\n\s*\n(?=(?:Terme d.indexation pour les affaires|Sujet|Lieu|"
    r"Société / organisation|Identificateur / mot-clé|Classification|Titre):)")
FIELD = re.compile(r"^([^:\n]{1,80}):[ \t]*(.*(?:\n(?!\n|[^:\n]{1,80}:).+)*)", re.M)
MONTHS = {name: i for i, name in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
PUBLICATIONS = ("Financial Times; London (UK)", "Oil & Gas News; Manama", "Egypt Oil & Gas; Cairo")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    # Conservative cleanup: retain wording, punctuation, captions and bylines.
    return " ".join(unicodedata.normalize("NFC", text).split())


def parse_date(value):
    """Return null for unrecognised/partial dates rather than inventing a day."""
    if not value:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        match = re.fullmatch(r"([A-Za-z]{3}) (\d{1,2}), (\d{4})", value)
        if match and match[1] in MONTHS:
            try:
                return date(int(match[3]), MONTHS[match[1]], int(match[2])).isoformat()
            except ValueError:
                pass
    return None


def parse_record(block):
    start = BODY_START.search(block)
    if not start:
        raise ValueError("Missing full-text marker")
    boundary = METADATA_START.search(block, start.end())
    if not boundary:
        raise ValueError("Missing full-text/metadata boundary")
    body = block[start.end():boundary.start()].strip()
    metadata = {m[1].strip(): m[2].strip() for m in FIELD.finditer(block[boundary.end():])}
    required = ("Titre", "Publication")
    document_id_raw = metadata.get("ID de document ProQuest", metadata.get("IDde document ProQuest"))
    if not body or any(not metadata.get(key) for key in required) or not document_id_raw:
        raise ValueError("Missing body or required title/publication/document ID")
    document_id = document_id_raw
    flags = ["publication_timing_unverified", "query_provenance_not_recorded"]
    if "ID de document ProQuest" not in metadata:
        flags.append("document_id_label_variant")
    if not document_id.isdigit():
        # Correct observed export spacing only when the document URL corroborates it.
        compact = re.sub(r"\s+", "", document_id)
        url_ids = re.findall(r"/docview/(\d+)(?=[/?#]|$)", metadata.get("URL du document", ""))
        if not compact.isascii() or not compact.isdigit() or url_ids != [compact]:
            raise ValueError("Malformed document ID")
        document_id = compact
        flags.append("document_id_spacing_normalized_url_verified")
    raw_date = metadata.get("Date de publication")
    if raw_date is None and "Datede publication" in metadata:
        raw_date = metadata["Datede publication"]
        flags.append("publication_date_label_variant")
    published = parse_date(raw_date)
    if not published and raw_date:
        spaced = re.fullmatch(r"([A-Za-z]{3}) ([0-9]\s+[0-9]), (\d{4})", raw_date)
        copyright_date = re.search(r"([A-Za-z]{3} \d{1,2}, \d{4})$", metadata.get("Copyright", ""))
        if spaced and copyright_date:
            candidate = parse_date(f"{spaced[1]} {re.sub(r'\s+', '', spaced[2])}, {spaced[3]}")
            if candidate and candidate == parse_date(copyright_date[1]):
                published = candidate
                flags.append("publication_date_spacing_normalized_copyright_verified")
    updated = parse_date(metadata.get("Dernière mise à jour"))
    if not published:
        flags.append("publication_date_unparsed")
    else:
        d = date.fromisoformat(published)
        if d.day == calendar.monthrange(d.year, d.month)[1]:
            flags.append("month_end_date_review")
        if not "2019-01-01" <= published <= "2025-12-31":
            flags.append("outside_study_period")
    availability = metadata.get("Disponibilité du texte intégral")
    if not availability:
        flags.append("archive_availability_unknown")
    if availability and "30daysafter" in re.sub(r"\s+", "", availability):
        flags.append("archive_loading_delay_30_days")
        if "30 days after" not in availability:
            flags.append("availability_notice_spacing_normalized")
    if updated and published and updated > published:
        flags.append("record_updated_after_publication")
    if "\ufffd" in body or any(ord(c) < 32 and c not in "\n\t\r" for c in body):
        flags.append("text_encoding_review")
    clean = normalize(body)
    publication_raw = metadata["Publication"]
    publication_group = next((name for name in PUBLICATIONS
                              if re.sub(r"\s+", "", name) == re.sub(r"\s+", "", publication_raw)), None)
    if publication_group and publication_group != publication_raw:
        flags.append("publication_name_spacing_normalized")
    # Preserve metadata changes too; never silently pick the newest version.
    fingerprint = digest(json.dumps({"body": clean, "metadata": metadata},
                                    sort_keys=True, ensure_ascii=False))
    return {
        "document_id": document_id,
        "document_id_raw": document_id_raw,
        "record_version_hash": fingerprint,
        "title": metadata["Titre"],
        "body_raw": body,
        "body_clean": clean,
        "body_hash": digest(clean),
        "word_count": len(clean.split()),
        "publication_raw": publication_raw,
        "publication_group": publication_group,
        "authors": metadata.get("Auteur"),
        "publication_date_raw": raw_date,
        "publication_date": published,
        "date_precision": "unverified",
        "record_updated_at": updated,
        "availability_note": availability,
        "archive_available_at": None,
        "query_version": None,
        "metadata_raw": metadata,
        "quality_flags": flags,
        "provenance": [],
    }


def import_files(paths):
    imported_at = datetime.now(timezone.utc).isoformat()
    versions = {}
    files = []
    errors = []
    for path in sorted(paths):
        raw = path.read_bytes()
        # Normalise export line endings; the original file stays untouched.
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        file_hash = hashlib.sha256(raw).hexdigest()
        file_ids = []
        years = Counter()
        for number, block in enumerate(SEPARATOR.split(text), 1):
            if not BODY_START.search(block) and not re.search(r"^ID\s*de document ProQuest:", block, re.M):
                continue
            try:
                record = parse_record(block)
            except ValueError as exc:
                errors.append({"export_file": path.name, "block": number, "error": str(exc)})
                continue
            file_ids.append(record["document_id"])
            years[record["publication_date"][:4] if record["publication_date"] else "unknown"] += 1
            key = (record["document_id"], record["record_version_hash"])
            if key not in versions:
                record["imported_at"] = imported_at
                versions[key] = record
            versions[key]["provenance"].append({
                "export_file": path.name, "export_sha256": file_hash,
                "block": number, "query_version": None,
            })
        files.append({"export_file": path.name, "sha256": file_hash,
                      "parsed_records": len(file_ids), "unique_ids": len(set(file_ids)),
                      "year_counts": dict(sorted(years.items()))})
    records = sorted(versions.values(), key=lambda r: (
        r["publication_date"] or "", r["document_id"], r["record_version_hash"]))
    ids, bodies = defaultdict(list), defaultdict(set)
    for record in records:
        ids[record["document_id"]].append(record["record_version_hash"])
        bodies[record["body_hash"]].add(record["document_id"])
    report = {
        "imported_at": imported_at, "files": files,
        "parsed_occurrences": sum(f["parsed_records"] for f in files),
        "unique_ids": len(ids), "record_versions": len(records),
        "repeated_identical_versions": sum(f["parsed_records"] for f in files) - len(records),
        "unique_bodies": len(bodies),
        "ids_with_multiple_versions": {key: value for key, value in ids.items() if len(value) > 1},
        "identical_bodies_across_ids": {key: sorted(value) for key, value in bodies.items() if len(value) > 1},
        "source_counts_by_record_version": dict(Counter(r["publication_raw"] for r in records)),
        "source_group_counts_by_record_version": dict(Counter(r["publication_group"] or r["publication_raw"] for r in records)),
        "quality_flags_by_record_version": dict(Counter(f for r in records for f in r["quality_flags"])),
        "errors": errors,
        "limitations": ["Search completeness is not inferred from imported counts.",
                        "Dates and original historical versions remain unverified.",
                        "Exact-body checks do not identify paraphrases or the same underlying event.",
                        "No relevance labels or financial indicators have been inferred."],
    }
    return records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/raw/proquest")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/proquest_articles.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "data/audit/proquest_import.json")
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.txt"))
    if not paths:
        parser.error("No TXT exports found in input directory")
    protected = {p.resolve() for p in paths}
    if args.output.resolve() in protected or args.report.resolve() in protected or args.output.resolve() == args.report.resolve():
        parser.error("Outputs must be distinct and must not replace raw input files")
    records, report = import_files(paths)
    if report["errors"] or not records:
        for error in report["errors"]:
            print(f"{error['export_file']}, block {error['block']}: {error['error']}")
        parser.exit(1, "Import stopped; outputs unchanged. Review unsupported or incomplete records.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {report['parsed_occurrences']} occurrences from {len(paths)} files: "
          f"{report['unique_ids']} unique IDs, {report['record_versions']} preserved versions, "
          f"{report['repeated_identical_versions']} repeated identical versions.")
    print(f"Articles: {args.output}\nAudit: {args.report}")


if __name__ == "__main__":
    main()
