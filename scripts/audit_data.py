"""Small, dependency-free data-access audit. No model fitting or forecasting."""

import argparse
import calendar
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
START = "2019-01-01"
END = "2025-12-31"
QUERY = '(Hormuz OR "Red Sea" OR "Bab el-Mandeb" OR "Bab al-Mandab") AND (shipping OR ship OR tanker OR maritime OR oil)'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch(base, params=None):
    url = base + ("?" + urlencode(params) if params else "")
    request = Request(url, headers={"User-Agent": "MaritimeNewsResearch/0.1"})
    try:
        with urlopen(request, timeout=45) as response:
            return response.read().decode("utf-8-sig")
    except HTTPError as exc:
        # Never print request URLs: Guardian authentication is in the query string.
        raise RuntimeError(f"Source returned HTTP {exc.code}; check access and quota.") from None
    except TimeoutError:
        raise RuntimeError("Source request timed out; retry the audit later.") from None
    except URLError:
        raise RuntimeError("Source connection failed; check network access.") from None


def guardian_key():
    key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "GUARDIAN_API_KEY":
                key = value.strip().strip("\"'")
                break
    if not key:
        raise RuntimeError("Add GUARDIAN_API_KEY to the local .env file first.")
    return key


def audit_ovx(input_file=None):
    source = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    audited = stamp()
    text = (Path(input_file).read_text(encoding="utf-8-sig") if input_file
            else fetch(source, {"id": "OVXCLS", "cosd": START, "coed": END}))
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "OVXCLS" not in reader.fieldnames:
        raise RuntimeError("Unexpected OVX response: expected a CSV with OVXCLS.")
    date_column = reader.fieldnames[0]
    rows = [row for row in reader if START <= row[date_column] <= END]
    if not rows:
        raise RuntimeError("OVX source returned no observations in the study period.")
    values, missing = [], []
    for row in rows:
        try:
            value = float(row["OVXCLS"])
        except ValueError:
            missing.append(row[date_column])
            continue
        if not math.isfinite(value):
            missing.append(row[date_column])
            continue
        values.append((row[date_column], value))
    if not values:
        raise RuntimeError("OVX source returned no numeric observations.")
    dates = [row[date_column] for row in rows]
    path = ROOT / "data/raw/ovx_2019_2025.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    summary = {
        "source": source, "series": "OVXCLS", "audited_at": audited,
        "retrieved_at": None if input_file else audited,
        "local_import": str(Path(input_file).resolve()) if input_file else None,
        "requested_start": START, "requested_end": END,
        "first_numeric_date": min(date for date, _ in values),
        "last_numeric_date": max(date for date, _ in values),
        "rows_in_period": len(rows), "numeric_observations": len(values),
        "missing_value_dates": missing, "duplicate_dates": len(dates) - len(set(dates)),
        "min": min(value for _, value in values), "max": max(value for _, value in values),
        "notes": ["No imputation performed. Holidays may explain missing values.",
                  "Dates absent from the source require a separate calendar audit.",
                  "Retrieved history is not a point-in-time data vintage."],
    }
    save_json("data/audit/ovx_summary.json", summary)
    print(f"OVX: {len(values)} numeric observations; {len(missing)} missing rows.")
    print("Saved data/audit/ovx_summary.json")


def audit_guardian(monthly=False):
    key = guardian_key()
    periods = [(2019, 6), (2021, 3), (2023, 12), (2025, 6)]
    if monthly:
        periods = [(year, month) for year in range(2019, 2026) for month in range(1, 13)]
    rows = []
    for year, month in periods:
        month_id = f"{year}-{month:02d}"
        path = ROOT / f"data/raw/guardian/{month_id}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("query") != QUERY:
                raise RuntimeError("Cached query differs; use a separate cache for a new query.")
        else:
            raw = json.loads(fetch("https://content.guardianapis.com/search", {
                "api-key": key, "q": QUERY,
                "from-date": f"{month_id}-01",
                "to-date": f"{month_id}-{calendar.monthrange(year, month)[1]}",
                "page-size": 5, "order-by": "oldest", "type": "article",
                "show-fields": "bodyText,lastModified",
            }))
            response = raw.get("response", {})
            if response.get("status") != "ok":
                raise RuntimeError("Guardian returned an unsuccessful response.")
            payload = {"retrieved_at": stamp(), "query": QUERY, "response": response}
            save_json(path.relative_to(ROOT), payload)
            time.sleep(1.1)
        response = payload["response"]
        articles = response.get("results", [])
        row = {
            "month": month_id, "total_matches": response["total"],
            "sample_size": len(articles),
            "samples_with_body": sum(bool(a.get("fields", {}).get("bodyText")) for a in articles),
            "samples_with_update_date": sum(bool(a.get("fields", {}).get("lastModified")) for a in articles),
        }
        rows.append(row)
        print(f"{month_id}: {row['total_matches']} matches, {row['samples_with_body']}/{len(articles)} sampled bodies.")
    save_json("data/audit/guardian_summary.json", {
        "created_at": stamp(), "query": QUERY, "monthly_audit": monthly, "periods": rows,
        "notes": ["Counts are query matches, not verified relevant events.",
                  "Only the first five chronological articles per month are cached.",
                  "This convenience sample is for access checks, not classifier evaluation.",
                  "Current article bodies may differ from their original publication versions."],
    })
    print("Saved data/audit/guardian_summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["ovx", "guardian"])
    parser.add_argument("--monthly", action="store_true", help="Audit all 84 months for Guardian.")
    parser.add_argument("--file", type=Path, help="Audit a manually downloaded FRED OVX CSV.")
    args = parser.parse_args()
    if args.file and args.source != "ovx":
        parser.error("--file is available only for OVX.")
    try:
        audit_ovx(args.file) if args.source == "ovx" else audit_guardian(args.monthly)
    except (RuntimeError, ValueError, OSError) as exc:
        # Avoid uncaught response/error details that could disclose authentication.
        if isinstance(exc, RuntimeError):
            parser.exit(1, f"Audit stopped: {exc}\n")
        parser.exit(1, f"Audit stopped: data parsing or storage failed ({type(exc).__name__}).\n")


if __name__ == "__main__":
    main()
