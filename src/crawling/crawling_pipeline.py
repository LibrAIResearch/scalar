import json
import os
from crawl_data import crawl_arxiv_paper
from get_citations import get_bibitems
from augment_arxiv_link import update_citations
import asyncio
import re
from tqdm import tqdm
from asyncio import Semaphore
import time
from datetime import datetime, timedelta
import logging

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("crawling.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add these constants at the top level
MAX_REQUESTS_PER_MINUTE = 80
RATE_LIMIT_WINDOW = 60  # seconds
MAX_CONCURRENT_REQUESTS = 20


def get_paper_list(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def is_valid_arxiv_id(arxiv_id):
    # Basic validation for arXiv ID format
    if not arxiv_id:
        return False
    # Check if it matches common arXiv ID patterns (e.g., 2301.00000 or math/0000000)
    patterns = [
        r"^\d{4}\.\d{4,5}(v\d+)?$",  # New format: YYMM.number
        r"^[a-zA-Z-]+/\d{7}(v\d+)?$",  # Old format: category/YYMMNNN
    ]
    return any(re.match(pattern, arxiv_id) for pattern in patterns)


async def process_citation(citation, data_dir, arxiv_id, semaphore, rate_limiter):
    citation_arxiv_id = citation["arxiv_id"]
    if citation_arxiv_id and is_valid_arxiv_id(citation_arxiv_id):
        async with semaphore:
            try:
                await rate_limiter()
                logger.debug(
                    f"Processing citation {citation_arxiv_id} for paper {arxiv_id}"
                )
                paper_data = crawl_arxiv_paper(citation_arxiv_id)

                # Create citations directory if it doesn't exist
                citations_dir = os.path.join(data_dir, arxiv_id, "citations")
                os.makedirs(citations_dir, exist_ok=True)

                try:
                    with open(
                        os.path.join(citations_dir, f"{citation_arxiv_id}.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(paper_data, f)
                    logger.debug(f"Successfully processed citation {citation_arxiv_id}")
                except OSError as e:
                    logger.error(
                        f"Error writing citation file for {citation_arxiv_id}: {str(e)}"
                    )

            except Exception as e:
                logger.error(f"Error processing citation {citation_arxiv_id}: {str(e)}")
                raise
    else:
        logger.warning(f"Invalid or missing arxiv id for citation: {citation}")


async def crawl_paper(paper_list, data_dir):
    # Main progress bar for papers
    paper_pbar = tqdm(paper_list, desc="Processing papers", position=0)

    # Create shared rate limiter state
    request_times = []

    async def rate_limiter():
        now = datetime.now()
        # Remove requests older than the window
        while request_times and now - request_times[0] > timedelta(
            seconds=RATE_LIMIT_WINDOW
        ):
            request_times.pop(0)
        # If at rate limit, wait until oldest request expires
        if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
            wait_time = (
                request_times[0] + timedelta(seconds=RATE_LIMIT_WINDOW) - now
            ).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        request_times.append(now)

    async def process_single_paper(paper):
        arxiv_id = paper["arxiv_id"]
        try:
            paper_pbar.set_description(f"Processing paper {arxiv_id}")
            logger.info(f"Starting to process paper {arxiv_id}")

            # Create paper directory
            paper_dir = os.path.join(data_dir, arxiv_id)
            os.makedirs(paper_dir, exist_ok=True)
            logger.debug(f"Created directory for paper {arxiv_id}")

            # Step 1: Get paper data if not exists
            paper_data_path = os.path.join(paper_dir, "paper_data.json")
            if not os.path.exists(paper_data_path):
                await rate_limiter()  # Rate limit the initial paper fetch
                paper_data = crawl_arxiv_paper(arxiv_id)
                with open(paper_data_path, "w", encoding="utf-8") as f:
                    json.dump(paper_data, f)
                logger.debug(f"Saved paper data for {arxiv_id}")
            else:
                with open(paper_data_path, "r", encoding="utf-8") as f:
                    paper_data = json.load(f)
                logger.debug(f"Loaded existing paper data for {arxiv_id}")

            # Step 2: Get citations if not exists
            citations_path = os.path.join(paper_dir, "paper_citations.json")
            if not os.path.exists(citations_path):
                citations = get_bibitems(paper_data["html"])
                with open(citations_path, "w", encoding="utf-8") as f:
                    json.dump(citations, f)
                logger.debug(f"Saved {len(citations)} citations for paper {arxiv_id}")
            else:
                with open(citations_path, "r", encoding="utf-8") as f:
                    citations = json.load(f)
                logger.debug(f"Loaded existing citations for paper {arxiv_id}")

            # Step 3: Get augmented citations if not exists
            augmented_citations_path = os.path.join(
                paper_dir, "augmented_paper_citations.json"
            )
            if not os.path.exists(augmented_citations_path):
            # if True:
                updated_citations = await update_citations(citations)
                with open(augmented_citations_path, "w", encoding="utf-8") as f:
                    json.dump(updated_citations, f)
                logger.debug(f"Saved augmented citations for paper {arxiv_id}")
            else:
                with open(augmented_citations_path, "r", encoding="utf-8") as f:
                    updated_citations = json.load(f)
                logger.debug(
                    f"Loaded existing augmented citations for paper {arxiv_id}"
                )

            # Step 4: Save metadata if not exists
            metadata_path = os.path.join(paper_dir, "paper_metadata.json")
            if not os.path.exists(metadata_path):
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(paper, f)
                logger.debug(f"Saved metadata for paper {arxiv_id}")

            # Step 5: Process citations
            citations_dir = os.path.join(paper_dir, "citations")
            os.makedirs(citations_dir, exist_ok=True)

            # Only process citations that don't already exist
            citations_to_process = []
            for citation in updated_citations:
                citation_arxiv_id = citation.get("arxiv_id")
                if citation_arxiv_id and is_valid_arxiv_id(citation_arxiv_id):
                    citation_path = os.path.join(
                        citations_dir, f"{citation_arxiv_id}.json"
                    )
                    if not os.path.exists(citation_path):
                        citations_to_process.append(citation)

            if citations_to_process:
                semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
                citation_pbar = tqdm(
                    total=len(citations_to_process),
                    desc=f"Citations for {arxiv_id}",
                    position=1,
                    leave=False,
                )

                async def process_citation_with_progress(citation):
                    try:
                        await process_citation(
                            citation, data_dir, arxiv_id, semaphore, rate_limiter
                        )
                        citation_pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing citation: {str(e)}")
                        raise

                tasks = [
                    process_citation_with_progress(citation)
                    for citation in citations_to_process
                ]
                await asyncio.gather(*tasks)
                citation_pbar.close()
                logger.info(
                    f"Successfully processed {len(citations_to_process)} new citations for paper {arxiv_id}"
                )
            else:
                logger.info(f"No new citations to process for paper {arxiv_id}")

        except Exception as e:
            logger.error(f"Error processing paper {arxiv_id}: {str(e)}")
            raise

    # Process papers concurrently
    tasks = [process_single_paper(paper) for paper in paper_list]
    await asyncio.gather(*tasks)
    paper_pbar.close()


if __name__ == "__main__":
    paper_list = get_paper_list("./sample_data/icl2025_metadata.json")
    logger.info(f"Starting crawling process for {len(paper_list)} papers")
    asyncio.run(crawl_paper(paper_list[:1], "./sample_data"))
    logger.info("Crawling process completed successfully")