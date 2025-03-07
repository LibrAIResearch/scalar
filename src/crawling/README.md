# Academic Paper Crawling Pipeline

This project implements a pipeline for crawling academic papers from arXiv and extracting their citations. The pipeline handles rate limiting, parallel processing, and persistent storage of paper data.

## Overview

The pipeline consists of several key components:

- `crawling_pipeline.py`: Main pipeline orchestrator that coordinates the crawling process
- `crawl_data.py`: Handles paper data retrieval from ar5iv.org using Firecrawl API
- `get_citations.py`: Extracts citation information from paper HTML content
- `augment_arxiv_link.py`: Enriches citation data by finding arXiv IDs using search APIs

## Requirements

- Python 3.7+
- Required environment variables:
  - `FIRECRAWL_API_KEY`: API key for Firecrawl service
  - `SERPER_API_KEY`: API key for Google Serper API

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare input data:

   - Create a JSON file containing paper metadata with arXiv IDs
   - Example format: `[{"arxiv_id": "2301.00000", ...}, ...]`

2. Run the pipeline:

```bash
python crawling_pipeline.py
```

## Pipeline Steps

1. **Paper Data Collection**

   - Fetches paper HTML content from ar5iv.org
   - Stores raw paper data in JSON format

2. **Citation Extraction**

   - Parses HTML to extract bibliography items
   - Extracts metadata like titles, authors, venues

3. **Citation Enrichment**

   - Attempts to find arXiv IDs for citations using search APIs
   - Handles rate limiting and parallel processing

4. **Data Storage**
   - Creates directory structure: `data_dir/arxiv_id/`
   - Stores multiple JSON files per paper:
     - `paper_data.json`: Raw paper data
     - `paper_citations.json`: Extracted citations
     - `augmented_paper_citations.json`: Enriched citations
     - `citations/*.json`: Individual citation paper data

## Rate Limiting

- Maximum 80 requests per minute
- Maximum 20 concurrent requests
- Automatic rate limiting and backoff

## Logging

- Logs are written to `crawling.log`
- Includes INFO and ERROR level messages
- Console output shows progress bars

## Error Handling

- Graceful handling of API failures
- Skips existing files for resume capability
- Detailed error logging for debugging
