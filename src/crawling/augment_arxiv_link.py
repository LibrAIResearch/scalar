import json
import re
import aiohttp
import asyncio
from tqdm import tqdm
from difflib import SequenceMatcher
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def string_similarity(a, b):
    """
    Calculate similarity ratio between two strings using SequenceMatcher.

    Args:
        a (str): First string to compare
        b (str): Second string to compare

    Returns:
        float: Similarity ratio between 0 and 1
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def extract_arxiv_id_from_venue(venue):
    # Extract arxiv ID from venues like "arXiv preprint arXiv:2303.08774"
    match = re.search(r"arXiv:(\d+\.\d+)", venue)
    if match:
        return match.group(1)
    return None


async def search_arxiv_api_batch(session, citations, batch_size=100):
    url = "https://google.serper.dev/search"
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable is not set")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    results = {}
    try:
        # Prepare batch of queries
        queries = []
        citation_map = {}  # Map query to citation index

        for i, citation in enumerate(citations):
            if citation["title"]:
                query = f"arxiv.org/abs {citation['title']}"
                queries.append({"q": query, "num": 10})
                citation_map[query] = i

        # Process in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            payload = json.dumps(batch)

            async with session.post(url, headers=headers, data=payload) as response:
                await asyncio.sleep(0.1)  # Rate limiting

                if response.status == 200:
                    batch_results = await response.json()

                    # Process each result in the batch
                    for query, result in zip(batch, batch_results):
                        citation_idx = citation_map[query["q"]]

                        for organic in result.get("organic", []):
                            link = organic["link"]
                            if "arxiv.org/abs/" in link:
                                # Extract arxiv ID and clean title from search result
                                search_title = organic["title"]
                                # Remove arxiv ID prefix if exists
                                clean_search_title = re.sub(
                                    r"^\[\d+\.\d+\]\s*", "", search_title
                                )
                                # Remove "- arXiv" suffix if exists
                                clean_search_title = re.sub(
                                    r"\s*-\s*arXiv$", "", clean_search_title
                                )
                                # Extract arxiv ID from URL
                                arxiv_match = re.search(
                                    r"arxiv\.org/abs/(\d+\.\d+)", link
                                )
                                arxiv_id = arxiv_match.group(1) if arxiv_match else None

                                # print(
                                #     "#" * 20,
                                #     "\n",
                                #     query["q"][14:],
                                #     "\n",
                                #     link,
                                #     "\n",
                                #     clean_search_title,
                                #     "\n",
                                #     string_similarity(
                                #         query["q"][14:], clean_search_title
                                #     ),
                                #     "\n",
                                # )
                                if (
                                    string_similarity(
                                        query["q"][14:], clean_search_title
                                    )
                                    > 0.7
                                ):
                                    if arxiv_id:
                                        results[citation_idx] = arxiv_id
                                        # print(
                                        #     f"Updated citation {citation_idx} with arxiv ID {arxiv_id}"
                                        # )
                                        break

    except Exception as e:
        print(f"Error in batch search: {e}")

    return results


async def process_citations_batch(session, citations):
    # Then batch search remaining citations without arxiv IDs
    citations_to_search = [c for c in citations]
    if citations_to_search:
        search_results = await search_arxiv_api_batch(session, citations_to_search)

        # Update citations with search results
        for idx, arxiv_id in search_results.items():
            citations_to_search[idx]["url"] = f"https://arxiv.org/abs/{arxiv_id}"
            citations_to_search[idx]["arxiv_id"] = arxiv_id

    # First try to extract from venues
    for citation in citations:
        if citation["url"] and "arxiv.org/abs/" in citation["url"]:
            continue

        if citation["venue"]:
            arxiv_id = extract_arxiv_id_from_venue(citation["venue"])
            if arxiv_id:
                citation["url"] = f"https://arxiv.org/abs/{arxiv_id}"
                citation["arxiv_id"] = arxiv_id

    return citations


async def update_citations(citations):
    async with aiohttp.ClientSession() as session:
        results = await process_citations_batch(session, citations)

        # Show progress bar after processing
        for _ in tqdm(range(len(results)), desc="Processing citations"):
            pass

    return results

