# Install with pip install firecrawl-py
from firecrawl import FirecrawlApp
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def crawl_arxiv_paper(paper_id: str) -> dict:
    """
    Crawl an arXiv paper using Firecrawl API.

    Args:
        paper_id (str): The arXiv paper ID (e.g., '2411.11843')

    Returns:
        dict: The crawled paper data containing markdown and HTML formats
    """
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable is not set")

    app = FirecrawlApp(api_key=api_key)

    url = f"https://ar5iv.org/html/{paper_id}"
    response = app.scrape_url(
        url=url,
        params={
            "formats": ["markdown", "html"],
        },
    )

    return response
