"""
Downloads sample public financial documents to get started quickly.
These are public documents from official sources (RBI, company IR pages).

Usage:
    python scripts/download_sample_docs.py
"""

import urllib.request
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Public financial documents (URLs may change over time - if one fails,
# manually download any annual report / RBI circular PDF instead)
SAMPLES = {
    # SEC investor-education guide to reading a 10-K (public domain, stable URL)
    "SEC_How_to_Read_a_10-K.pdf":
        "https://www.sec.gov/files/reada10k.pdf",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def download(name: str, url: str):
    dest = DOCS_DIR / name
    if dest.exists():
        print(f"Already exists: {name}")
        return
    print(f"Downloading {name} ...")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            # A dead/redirected URL often still returns HTTP 200 with an HTML
            # error page -- fail loudly here instead of silently writing garbage
            # that only surfaces as a cryptic pypdf crash during ingest.
            raise ValueError(f"response is not a PDF (got {data[:20]!r})")
        dest.write_bytes(data)
        print(f"  Saved to {dest}")
    except Exception as e:
        print(f"  Failed: {e}")
        print(f"  -> Manually download a PDF and place it in {DOCS_DIR}")


if __name__ == "__main__":
    for name, url in SAMPLES.items():
        download(name, url)
    print("\nTip: also grab annual reports from company investor relations pages:")
    print("  - HDFC Bank: hdfcbank.com -> Investor Relations -> Annual Reports")
    print("  - ICICI: icicibank.com -> About Us -> Investor Relations")
    print("  - RBI circulars: rbi.org.in -> Notifications")
