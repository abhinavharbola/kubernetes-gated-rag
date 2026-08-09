"""
Fetches a list of doc pages into data/official/ or data/community/ for ingestion.

Run this yourself, from your own machine, not from a hosted/shared environment,
so the traffic is attributed to you and you're the one bound by each site's
terms of use and robots.txt.

Usage:
    pip install requests beautifulsoup4
    python fetch_docs.py

Edit OFFICIAL_URLS / COMMUNITY_URLS below to whatever you actually want.
Each page is saved as clean-ish text under data/<authority>/<slug>.txt,
which src/parsers.py already knows how to read via the .txt path.
"""

import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "terraform-agentic-rag-ingest/1.0 (personal project, contact: you@example.com)"
REQUEST_DELAY_SECONDS = 2.0  # be polite, don't hammer someone else's docs site
TIMEOUT_SECONDS = 15

OFFICIAL_URLS = [
    "https://developer.hashicorp.com/terraform/language",
    "https://developer.hashicorp.com/terraform/language/syntax",
    "https://developer.hashicorp.com/terraform/language/resources/syntax",
    "https://developer.hashicorp.com/terraform/language/values/variables",
    "https://developer.hashicorp.com/terraform/language/values/outputs",
    "https://developer.hashicorp.com/terraform/language/expressions",
    "https://developer.hashicorp.com/terraform/language/meta-arguments/count",
    "https://developer.hashicorp.com/terraform/language/meta-arguments/for_each",
    "https://developer.hashicorp.com/terraform/language/state",
    "https://developer.hashicorp.com/terraform/language/backend",
    "https://developer.hashicorp.com/terraform/language/modules",
    "https://developer.hashicorp.com/terraform/language/modules/develop/structure",
    "https://developer.hashicorp.com/terraform/language/providers/requirements",
    "https://developer.hashicorp.com/terraform/cli",
    "https://developer.hashicorp.com/terraform/cli/commands/plan",
    "https://developer.hashicorp.com/terraform/cli/commands/apply",
    "https://developer.hashicorp.com/terraform/cli/commands/destroy",
    "https://developer.hashicorp.com/terraform/cli/state",
]

COMMUNITY_URLS = [
    "https://registry.terraform.io/browse/modules",
    "https://blog.gruntwork.io/a-comprehensive-guide-to-terraform-b3d32832baca",
    "https://www.gruntwork.io/blog/how-to-create-reusable-infrastructure-with-terraform-modules",
    "https://www.gruntwork.io/blog/reusable-composable-battle-tested-terraform-modules",
    "https://docs.gruntwork.io/guides/style/terraform-style-guide/",
    "https://spacelift.io/blog/terraform-best-practices",
    "https://spacelift.io/blog/terraform-security",
    "https://spacelift.io/blog/terraform-state",
    "https://spacelift.io/blog/terraform-state-lock",
    "https://spacelift.io/blog/terraform-remote-state",
]


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower()
    return slug or "index"


def fetch_and_clean(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    # collapse repeated blank lines left over from stripped elements
    return re.sub(r"\n{3,}", "\n\n", text)


def fetch_all(urls: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        target = out_dir / f"{slugify(url)}.txt"
        if target.exists():
            print(f"skip (already fetched): {url}")
            continue
        try:
            text = fetch_and_clean(url)
        except requests.RequestException as exc:
            print(f"FAILED: {url} -> {exc}")
            continue
        target.write_text(f"Source: {url}\n\n{text}", encoding="utf-8")
        print(f"saved: {url} -> {target}")
        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    fetch_all(OFFICIAL_URLS, repo_root / "data" / "official")
    fetch_all(COMMUNITY_URLS, repo_root / "data" / "community")