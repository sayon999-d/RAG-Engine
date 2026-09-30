import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    url: str
    title: str
    content: str
    metadata: dict[str, Any]


def validate_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or parsed.scheme not in ["http", "https"]:
            return False, "Invalid URL scheme"
        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid hostname"
        try:
            ip = socket.gethostbyname(hostname)
        except OSError:
            return False, "Could not resolve hostname"
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return False, "Restricted local/private IP"
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"


def scrape_url(url: str, timeout: int = 30) -> ScrapedContent:
    try:
        import trafilatura
    except ImportError:
        raise ImportError(
            "trafilatura required for web scraping. Install with: pip install trafilatura"
        )

    is_valid, msg = validate_url(url)
    if not is_valid:
        raise ValueError(f"Security Warning: {msg}")

    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if not downloaded:
            raise ValueError("Failed to download content")

        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            output_format="markdown",
            url=url,
        )

        if not content or len(content.strip()) < 100:
            raise ValueError("No meaningful content extracted")

        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else urlparse(url).netloc

        return ScrapedContent(
            url=url,
            title=title,
            content=content,
            metadata={
                "source": url,
                "doc_type": "web",
                "title": title,
                "author": metadata.author if metadata else "",
                "date": metadata.date if metadata else "",
                "categories": metadata.categories if metadata else [],
                "tags": metadata.tags if metadata else [],
            },
        )
    except Exception as e:
        logger.error(f"Web scraping failed for {url}: {e}")
        raise


def scrape_multiple_urls(urls: list[str], timeout: int = 30) -> list[ScrapedContent]:
    results = []
    for url in urls:
        try:
            result = scrape_url(url, timeout)
            results.append(result)
            logger.info(f"Successfully scraped: {url}")
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            results.append(
                ScrapedContent(
                    url=url, title="", content="", metadata={"error": str(e)}
                )
            )
    return results
