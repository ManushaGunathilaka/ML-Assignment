"""
scrape.py - Properties.lk House Data Scraper
=============================================
Scrapes house-for-sale listings from properties.lk using its Supabase REST API.
Outputs a CSV file with structured house data.

Ethical Scraping Note:
    This script uses the public anonymous API key embedded in the properties.lk
    frontend JavaScript bundle. It accesses the same Supabase REST endpoint that
    the website itself calls. A 1-second delay is added between requests to avoid
    overwhelming the server. The data is used solely for academic/research purposes.

Usage:
    python scrape.py                                   # defaults: 5500 records
    python scrape.py --max_records 1000                # scrape 1000 records
    python scrape.py --output my_data.csv              # custom output path
    python scrape.py --max_records 2000 --output data/houses.csv
"""

import requests
import pandas as pd
import time
import argparse
import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase API Configuration
# ---------------------------------------------------------------------------
API_URL = "https://mizevmlalbxggxnhikep.supabase.co/rest/v1/Ads"
API_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1pemV2bWxhbGJ4Z2d4bmhpa2VwIiwi"
    "cm9sZSI6ImFub24iLCJpYXQiOjE3MDkwMzI4ODIsImV4cCI6MjAyNDYwODg4Mn0."
    "ULUSG_v0VxSouW7OQQusgOHgM8CFoEMwhf5hbKBM2tc"
)
HEADERS = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "accept-profile": "public",
    "Accept": "application/json",
}

# Fields to keep (API name -> CSV column)
FIELD_MAP = {
    "id":          "id",
    "title":       "title",
    "totalPrice":  "price",
    "pricePer":    "price_type",
    "negotiable":  "negotiable",
    "bedrooms":    "bedrooms",
    "bathrooms":   "bathrooms",
    "houseSize":   "house_size",
    "landSize":    "land_size",
    "address":     "address",
    "subcity":     "city",
    "district":    "district",
    "description": "description",
    "timeStamp":   "posted_date",
}

BATCH_SIZE = 100       # records per API call
DELAY_SECONDS = 1      # polite delay between requests
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def fetch_batch(offset: int, limit: int) -> list[dict]:
    """Fetch one batch of records with retry + back-off."""
    params = {
        "select": "*",
        "approved": "eq.true",
        "category": "ilike.%houseforsale%",
        "order": "id.desc",
        "offset": offset,
        "limit": limit,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                API_URL, headers=HEADERS, params=params, timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                log.info("  Retrying in %ds ...", wait)
                time.sleep(wait)
            else:
                log.error("Giving up on batch at offset %d.", offset)
                return []


def extract_fields(record: dict) -> dict:
    """Pick and rename the fields we care about."""
    return {csv_col: record.get(api_key) for api_key, csv_col in FIELD_MAP.items()}


def scrape(max_records: int, output_path: str) -> None:
    """Main scraping loop."""
    log.info("=" * 55)
    log.info("Properties.lk House Scraper")
    log.info("Target: %d records | Batch: %d", max_records, BATCH_SIZE)
    log.info("=" * 55)

    all_rows: list[dict] = []
    total_batches = (max_records + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(total_batches):
        offset = i * BATCH_SIZE
        limit = min(BATCH_SIZE, max_records - len(all_rows))

        log.info("[%d/%d] offset=%d  limit=%d ...", i + 1, total_batches, offset, limit)
        data = fetch_batch(offset, limit)

        if not data:
            log.warning("Empty response -- stopping early.")
            break

        all_rows.extend(extract_fields(r) for r in data)
        log.info("  collected %d / %d", len(all_rows), max_records)

        if len(all_rows) >= max_records:
            break
        time.sleep(DELAY_SECONDS)

    df = pd.DataFrame(all_rows[:max_records])
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("Saved %d records to %s", len(df), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scrape houses from properties.lk")
    parser.add_argument(
        "--max_records", type=int, default=5500, help="Number of records to scrape (default: 5500)"
    )
    parser.add_argument(
        "--output", type=str, default="houses_data.csv", help="Output CSV path (default: houses_data.csv)"
    )
    args = parser.parse_args()
    scrape(args.max_records, args.output)


if __name__ == "__main__":
    main()
