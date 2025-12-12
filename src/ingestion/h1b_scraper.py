from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
        "image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

RAW_DATA_PATH = Path("data/raw/h1b_pages.json")
CONTENT_SELECTORS = [
    "main",
    "[role=main]",
    "#block-uscis-content",
    ".region-content",
    ".layout-main",
    ".usa-layout-docs__main",
    "article",
    "body",
]


def _ensure_data_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clean_text(text: str) -> str:
    replacements = {
        "\xa0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    cleaned = " ".join(text.split())
    return cleaned.strip()


def _select_main_container(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node:
            return node
    return None


def _slugify_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    slug = path.replace("/", "_")
    if parsed.query:
        slug = f"{slug}_{parsed.query.replace('=', '_').replace('&', '_')}"
    return slug


def _extract_main_text(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    main = _select_main_container(soup)
    if not main:
        return {"title": soup.title.get_text(strip=True) if soup.title else "", "text": ""}

    for tag_name in ("nav", "footer", "script", "style", "form", "aside"):
        for tag in main.find_all(tag_name):
            tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    paragraphs = [
        p.get_text(" ", strip=True) for p in main.find_all(["p", "h1", "h2", "h3", "li"])
    ]
    text = "\n".join(filter(None, [_clean_text(p) for p in paragraphs]))
    return {"title": title, "text": text}


def scrape_uscis_page(url: str, offline_html: Optional[Path] = None) -> Dict[str, object]:
    """
    Scrape a single USCIS H1B page and return structured data.
    """

    LOGGER.info("Scraping %s", url)
    record: Dict[str, object] = {
        "url": url,
        "title": "",
        "content": "",
        "scraped_at": datetime.utcnow().isoformat(),
        "success": False,
    }

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        parsed = _extract_main_text(response.text)
        record["title"] = parsed.get("title", "")
        record["content"] = parsed.get("text", "")
        record["success"] = bool(record["content"])
        if not record["content"]:
            LOGGER.warning("No main content found for %s", url)
    except requests.RequestException as exc:
        LOGGER.error("Failed to scrape %s: %s", url, exc)
        record["error"] = str(exc)
        if offline_html and offline_html.exists():
            LOGGER.info("Loading offline HTML fallback for %s", url)
            html = offline_html.read_text(encoding="utf-8")
            parsed = _extract_main_text(html)
            record["title"] = parsed.get("title", record.get("title", ""))
            record["content"] = parsed.get("text", "")
            record["success"] = bool(record["content"])
            if record["success"]:
                record["offline_source"] = str(offline_html)
    return record


def _offline_path_for_url(url: str, offline_dir: Optional[Path]) -> Optional[Path]:
    if not offline_dir:
        return None
    offline_dir.mkdir(parents=True, exist_ok=True)
    return offline_dir / f"{_slugify_url(url)}.html"


def scrape_h1b_pages(
    urls: Iterable[Union[str, Sequence[str]]],
    offline_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """
    Scrape multiple USCIS pages with support for fallback URL variants.
    """

    results: List[Dict[str, object]] = []
    normalized_groups: List[List[str]] = []

    for group in urls:
        if isinstance(group, str):
            normalized_groups.append([group])
            continue
        normalized_groups.append([candidate for candidate in group if candidate])

    for group in normalized_groups:
        if not group:
            continue
        page_record: Optional[Dict[str, object]] = None
        for url in group:
            offline_path = _offline_path_for_url(url, offline_dir)
            attempts = 0
            while attempts < 3:
                attempts += 1
                page_record = scrape_uscis_page(url, offline_html=offline_path)
                if page_record.get("success"):
                    results.append(page_record)
                    break
                LOGGER.warning("Retry %s/%s for %s", attempts, 3, url)
                time.sleep(1)
            if page_record and page_record.get("success"):
                break
            LOGGER.info("Moving to next variant for %s", url)

        if page_record and not page_record.get("success"):
            LOGGER.error("All variants failed for %s", group[0])

        time.sleep(1)  # ensure respectful delay between URLs

    successful_records = [record for record in results if record.get("success")]
    if not successful_records:
        raise RuntimeError("Failed to scrape any USCIS H1B pages.")

    _ensure_data_dir(RAW_DATA_PATH)
    with RAW_DATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(successful_records, handle, ensure_ascii=False, indent=2)

    return successful_records


__all__ = ["scrape_uscis_page", "scrape_h1b_pages"]
