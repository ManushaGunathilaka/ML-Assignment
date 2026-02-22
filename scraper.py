"""
Properties.lk House Data Scraper
=================================
Scrapes 5,500 house-for-sale records from properties.lk via its Supabase REST API.
Saves the data to houses_data.csv.
"""

import requests
import pandas as pd
import time
import sys

# --- Supabase API Configuration ---
API_URL = "https://mizevmlalbxggxnhikep.supabase.co/rest/v1/Ads"
API_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1pemV2bWxhbGJ4Z2d4bmhpa2VwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDkwMzI4ODIsImV4cCI6MjAyNDYwODg4Mn0."
    "ULUSG_v0VxSouW7OQQusgOHgM8CFoEMwhf5hbKBM2tc"
)

HEADERS = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "accept-profile": "public",
    "Accept": "application/json",
}

# --- Scraping Parameters ---
TARGET_RECORDS = 5500
BATCH_SIZE = 100          # records per API call
DELAY_SECONDS = 1         # pause between requests
MAX_RETRIES = 5           # retries per failed request
OUTPUT_FILE = "houses_data.csv"

# --- Fields to extract (API field -> CSV column) ---
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


def fetch_batch(offset: int, limit: int) -> list[dict]:
    """Fetch a single batch of records from the API with retry logic."""
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
            resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  [!] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  [X] Giving up on batch at offset {offset}.")
                return []


def extract_fields(record: dict) -> dict:
    """Extract and rename relevant fields from a raw API record."""
    row = {}
    for api_field, csv_col in FIELD_MAP.items():
        row[csv_col] = record.get(api_field, None)
    return row


def main():
    print("=" * 60)
    print("  Properties.lk House Scraper")
    print(f"  Target: {TARGET_RECORDS} records  |  Batch size: {BATCH_SIZE}")
    print("=" * 60)

    all_records: list[dict] = []
    total_batches = (TARGET_RECORDS + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        offset = batch_num * BATCH_SIZE
        remaining = TARGET_RECORDS - len(all_records)
        limit = min(BATCH_SIZE, remaining)

        print(f"\n[{batch_num + 1}/{total_batches}] Fetching {limit} records "
              f"(offset {offset}) ...", end=" ")

        data = fetch_batch(offset, limit)
        if not data:
            print("No data returned -- stopping early.")
            break

        rows = [extract_fields(r) for r in data]
        all_records.extend(rows)

        collected = len(all_records)
        print(f"OK  Total so far: {collected}")

        if collected >= TARGET_RECORDS:
            break

        time.sleep(DELAY_SECONDS)

    # --- Save to CSV ---
    df = pd.DataFrame(all_records[:TARGET_RECORDS])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"  [OK] Done! Saved {len(df)} records to {OUTPUT_FILE}")
    print(f"  Columns: {list(df.columns)}")
    print("=" * 60)

    # Quick summary
    print(f"\n-- Data Summary --")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Price range: {df['price'].min():,.0f} - {df['price'].max():,.0f}")
    print(f"  Districts:   {df['district'].nunique()} unique")
    print(f"  Avg bedrooms: {df['bedrooms'].mean():.1f}")


if __name__ == "__main__":
    main()
