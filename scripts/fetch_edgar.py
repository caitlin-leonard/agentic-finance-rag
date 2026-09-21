"""Fetch real 10-K filings from SEC EDGAR and emit the project's JSONL schema.

Usage:
    python scripts/fetch_edgar.py --ticker AAPL --year 2023 --out data/aapl_2023.jsonl

This produces records identical in shape to data/sample_filings.jsonl, so the
same ingestion path indexes real filings. SEC requires a descriptive
User-Agent; set EDGAR_UA="you@example.com" before running.

Note: full 10-K section parsing is non-trivial; this script pulls the primary
document and does a light section split. It's a starting point for real data,
not a production EDGAR parser.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

UA = os.getenv("EDGAR_UA", "agentic-finance-rag research (set EDGAR_UA)")
SECTION_PATTERNS = {
    "Item 1A - Risk Factors": r"item\s*1a\.?\s*risk factors",
    "Item 7 - MD&A": r"item\s*7\.?\s*management.s discussion",
    "Item 8 - Financial Statements": r"item\s*8\.?\s*financial statements",
}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def _cik_for(ticker: str) -> str:
    data = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise SystemExit(f"ticker {ticker} not found")


def fetch(ticker: str, year: int) -> list[dict]:
    cik = _cik_for(ticker)
    subs = json.loads(_get(
        f"https://data.sec.gov/submissions/CIK{cik}.json"))
    recent = subs["filings"]["recent"]
    accn = None
    for form, date, acc, doc in zip(
        recent["form"], recent["filingDate"],
        recent["accessionNumber"], recent["primaryDocument"],
    ):
        if form == "10-K" and date.startswith(str(year)):
            accn, primary = acc.replace("-", ""), doc
            break
    if not accn:
        raise SystemExit(f"no 10-K for {ticker} in {year}")
    url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/{primary}")
    html = _get(url)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)

    records = []
    low = text.lower()
    marks = sorted(
        [(m.start(), sec) for sec, pat in SECTION_PATTERNS.items()
         for m in [re.search(pat, low)] if m])
    for i, (pos, sec) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else min(pos + 20000, len(text))
        body = text[pos:end][:20000]
        records.append({
            "ticker": ticker.upper(), "company": subs["name"],
            "year": year, "section": sec, "kind": "prose", "text": body,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = fetch(args.ticker, args.year)
    with open(args.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} section records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
