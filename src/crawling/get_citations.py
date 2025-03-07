from typing import Optional
import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import unicodedata


def extract_urls(soup_item):
    """
    Extract a single URL from a BeautifulSoup item, prioritizing arXiv URLs
    Returns a single URL string or empty string if none found
    """
    arxiv_url = None
    first_url = None

    for a in soup_item.find_all("a", href=True):
        url = a["href"]
        if not first_url:
            first_url = url
        if "arxiv" in url.lower():
            arxiv_url = url
            break

    return arxiv_url or first_url or ""


def extract_arxiv_id(url: Optional[str]) -> Optional[str]:
    """Extract arXiv ID from URL"""
    if not url:
        return None
    patterns = [
        r"(\d+\.\d+)",
        r"([\w\-\.]+/\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def parse_bibitem(item):
    """
    Parse a bibliography item into structured data

    Args:
        item: BeautifulSoup object representing a bibliography item

    Returns:
        dict: Structured bibliography data containing:
            - id: bibliography ID
            - citation: short form citation (e.g. "Gu et al. (2021b)")
            - authors: list of author names
            - year: publication year
            - title: paper title
            - venue: publication venue
            - pages: page numbers (if available)
            - url: URL
    """
    # Initialize result dictionary
    result = {
        "id": item.get("id", ""),
        "citation": "",
        "authors": "",
        "year": "",
        "title": "",
        "venue": "",
        "pages": "",
        "url": extract_urls(item),
    }

    # Get citation tag (usually first span with ltx_tag class)
    citation_tag = item.find("span", class_="ltx_tag_bibitem")
    if citation_tag:
        result["citation"] = normalize_text(citation_tag.get_text(strip=True))

    # Process bibblocks (main content sections)
    bibblocks = item.find_all("span", class_="ltx_bibblock")

    for i, block in enumerate(bibblocks):
        block_text = normalize_text(block.get_text(strip=True))

        # First bibblock usually contains authors and year
        if i == 0:
            # Split on year pattern to separate authors and year
            year_match = re.search(r"\s*(19|20)\d{2}[a-z]?\.?\s*$", block_text)
            if year_match:
                authors_text = block_text[: year_match.start()]
                result["authors"] = normalize_text(authors_text.strip())
                result["year"] = normalize_text(year_match.group().strip(" ."))

        # Second bibblock typically contains the title
        elif i == 1 and not result["title"]:
            result["title"] = normalize_text(block_text)

        # Look for venue information (usually in italics)
        venue_italic = block.find("em")
        if venue_italic:
            venue_text = venue_italic.get_text(strip=True)
            result["venue"] = normalize_text(venue_text)

            # Try to extract page numbers after venue
            pages_match = re.search(r"pages?\s*(\d+[-–]\d+|\d+)", block_text)
            if pages_match:
                result["pages"] = normalize_text(pages_match.group(1))

    return result


def normalize_text(text: str) -> str:
    """
    Normalize text using Unicode NFKD normalization and ASCII encoding

    Args:
        text (str): Input text to normalize

    Returns:
        str: Normalized ASCII text
    """
    # Normalize unicode characters (decompose)
    normalized = unicodedata.normalize("NFKD", text)

    # Encode to ASCII, ignoring non-ASCII characters, then decode back to string
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")

    # Replace multiple spaces with single space
    ascii_text = " ".join(ascii_text.split())

    return ascii_text.strip()


def get_bibitems(html_content):
    try:
        # Send HTTP GET request
        # response = requests.get(url)
        # response.raise_for_status()

        # Create BeautifulSoup object
        soup = BeautifulSoup(html_content, "html.parser")

        # Find all elements with ltx_bibitem class
        bibitems = soup.find_all(class_="ltx_bibitem")

        # Extract text and URLs
        results = []
        for item in tqdm(bibitems):
            # Parse detailed bibliography information
            result = parse_bibitem(item)

            # Process URL for arxiv_id (now dealing with a single string)
            result["arxiv_id"] = extract_arxiv_id(result["url"])
            result["doi"] = None

            results.append(result)

        return results

    except requests.RequestException as e:
        print(f"Request error: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
