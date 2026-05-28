"""Download legal data from legal-ai-agent repository."""

import json
import urllib.request
from pathlib import Path

from src.common.config import RAW_DIR, QA_DATA_PATH, LEGAL_DATA_PATH


URLS = {
    "yuiTC_sample.json": "https://github.com/Paparusi/legal-ai-agent/raw/main/data/yuiTC_sample.json",
    "uts_vlc_processed.json": "https://github.com/Paparusi/legal-ai-agent/raw/main/data/uts_vlc_processed.json",
}


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress display."""
    print(f"Downloading {desc or dest.name}...")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB -> {dest}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in URLS.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"Already exists: {dest}")
            continue
        download_file(url, dest, filename)

    # Verify
    print("\n--- Verification ---")
    for filename in URLS:
        dest = RAW_DIR / filename
        if dest.exists():
            if filename.endswith(".json"):
                with open(dest, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "count" in data:
                    print(f"  {filename}: {data['count']} entries")
                elif isinstance(data, list):
                    print(f"  {filename}: {len(data)} entries")
                else:
                    print(f"  {filename}: loaded OK")
            else:
                print(f"  {filename}: {dest.stat().st_size / 1024:.0f} KB")
        else:
            print(f"  {filename}: MISSING")


if __name__ == "__main__":
    main()