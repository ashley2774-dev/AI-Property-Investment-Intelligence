from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("privateproperty_scraper")


@dataclass
class ScraperConfig:
    max_sale_records: int = 300
    max_rental_records: int = 300
    max_pages_per_seed: int = 20
    request_timeout: int = 30
    min_delay_seconds: float = 1.5
    max_delay_seconds: float = 3.0
    detail_pause_every: int = 40
    long_pause_seconds: float = 8.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )


SEARCH_SEEDS = {
    "sale": [
        {"property_type_seed": "House", "seed_name": "Johannesburg Metro Houses For Sale", "seed_url": "https://www.privateproperty.co.za/houses-for-sale/johannesburg-metro/33"},
        {"property_type_seed": "Apartment", "seed_name": "Johannesburg Metro Apartments For Sale", "seed_url": "https://www.privateproperty.co.za/apartments-for-sale/johannesburg-metro/33"},
        {"property_type_seed": "House", "seed_name": "Pretoria Houses For Sale", "seed_url": "https://www.privateproperty.co.za/houses-for-sale/pretoria/28"},
        {"property_type_seed": "Apartment", "seed_name": "Pretoria Apartments For Sale", "seed_url": "https://www.privateproperty.co.za/apartments-for-sale/pretoria/28"},
        {"property_type_seed": "House", "seed_name": "Centurion Houses For Sale", "seed_url": "https://www.privateproperty.co.za/houses-for-sale/centurion/32"},
        {"property_type_seed": "Apartment", "seed_name": "Centurion Apartments For Sale", "seed_url": "https://www.privateproperty.co.za/apartments-for-sale/centurion/32"},
    ],
    "rent": [
        {"property_type_seed": "House", "seed_name": "Johannesburg Metro Houses To Rent", "seed_url": "https://www.privateproperty.co.za/houses-to-rent/johannesburg-metro/33"},
        {"property_type_seed": "Apartment", "seed_name": "Johannesburg Metro Apartments To Rent", "seed_url": "https://www.privateproperty.co.za/apartments-to-rent/johannesburg-metro/33"},
        {"property_type_seed": "House", "seed_name": "Pretoria Houses To Rent", "seed_url": "https://www.privateproperty.co.za/houses-to-rent/pretoria/28"},
        {"property_type_seed": "Apartment", "seed_name": "Pretoria Apartments To Rent", "seed_url": "https://www.privateproperty.co.za/apartments-to-rent/pretoria/28"},
        {"property_type_seed": "House", "seed_name": "Centurion Houses To Rent", "seed_url": "https://www.privateproperty.co.za/houses-to-rent/centurion/32"},
        {"property_type_seed": "Apartment", "seed_name": "Centurion Apartments To Rent", "seed_url": "https://www.privateproperty.co.za/apartments-to-rent/centurion/32"},
    ],
}

SALE_DETAIL_RE = re.compile(r"/for-sale/")
RENT_DETAIL_RE = re.compile(r"/to-rent/")

LISTINGS_COLS = [
    "source_site", "listing_id", "listing_url", "title", "purchase_price", "suburb", "city", "province",
    "property_type", "bedrooms", "bathrooms", "parking_spaces", "garage", "floor_area_sqm", "land_area_sqm",
    "levies", "rates_taxes", "description", "listing_date", "scraped_timestamp", "pp_transaction_slug",
    "pp_province_slug", "pp_metro_slug", "pp_city_slug", "pp_suburb_slug"
]

RENTALS_COLS = [
    "source_site", "listing_id", "rental_url", "title", "monthly_rent", "suburb", "city", "province",
    "property_type", "bedrooms", "bathrooms", "parking_spaces", "garage", "floor_area_sqm", "land_area_sqm",
    "levies", "rates_taxes", "furnished_flag", "availability_text", "description", "listing_date",
    "scraped_timestamp", "pp_transaction_slug", "pp_province_slug", "pp_metro_slug", "pp_city_slug",
    "pp_suburb_slug"
]


def jitter_sleep(min_seconds: float, max_seconds: float) -> None:
    time.sleep(random.uniform(min_seconds, max_seconds))


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    value = value.replace("\xa0", " ").replace("\u202f", " ").replace("\u2009", " ").replace("\u2007", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def title_case_slug(value: Optional[str]) -> Optional[str]:
    value = clean_text(value)
    if value is None:
        return None
    return value.replace("-", " ").replace("_", " ").title()


def parse_float(value: Optional[str]) -> Optional[float]:
    value = clean_text(value)
    if value is None:
        return None
    value = value.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_int(value: Optional[str]) -> Optional[int]:
    parsed = parse_float(value)
    return None if parsed is None else int(round(parsed))


def parse_money(value: Optional[str]) -> Optional[float]:
    value = clean_text(value)
    if value is None:
        return None
    raw = value
    patterns = [
        r"R\s*([0-9][0-9\s,\.]{3,})",
        r"([0-9]{1,3}(?:[\s,][0-9]{3})+(?:\.\d+)?)",
        r"([0-9]+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.I)
        if not match:
            continue
        candidate = re.sub(r"[^0-9.]", "", match.group(1))
        if not candidate:
            continue
        try:
            amount = float(candidate)
            if amount > 0:
                return amount
        except Exception:
            continue
    return None


def first_non_null(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and clean_text(value) is None:
            continue
        return value
    return None


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    retries = Retry(total=4, connect=4, read=4, backoff_factor=1.2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-ZA,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return session


def fetch_html(session: requests.Session, url: str, cfg: ScraperConfig) -> str:
    response = session.get(url, timeout=cfg.request_timeout)
    response.raise_for_status()
    jitter_sleep(cfg.min_delay_seconds, cfg.max_delay_seconds)
    return response.text


def build_results_url(seed_url: str, page: int) -> str:
    return seed_url if page == 1 else f"{seed_url}?page={page}"


def extract_jsonld_objects(soup: BeautifulSoup) -> list[dict]:
    objects = []
    for script in soup.find_all("script", type=lambda x: x and "ld+json" in str(x)):
        raw = script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                objects.append(obj)
            elif isinstance(obj, list):
                objects.extend([x for x in obj if isinstance(x, dict)])
        except Exception:
            continue
    return objects


def extract_result_links(html: str, mode: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    pattern = SALE_DETAIL_RE if mode == "sale" else RENT_DETAIL_RE

    for obj in extract_jsonld_objects(soup):
        if obj.get("@type") != "Residence":
            continue
        url = clean_text(obj.get("url"))
        if url:
            urls.append(url.split("?")[0].rstrip("/"))

    for anchor in soup.find_all("a", href=True):
        href = clean_text(anchor["href"])
        if href is None:
            continue
        href = href.split("?")[0].rstrip("/")
        if href.startswith("/"):
            href = "https://www.privateproperty.co.za" + href
        if pattern.search(href):
            urls.append(href)

    return unique_preserve_order(urls)


def collect_detail_urls_for_mode(mode: str, cfg: ScraperConfig) -> list[str]:
    target = cfg.max_sale_records if mode == "sale" else cfg.max_rental_records
    session = build_session(cfg.user_agent)
    all_urls: list[str] = []
    seen: set[str] = set()

    for seed in SEARCH_SEEDS[mode]:
        if len(all_urls) >= target:
            break
        logger.info("%s seed: %s", mode.upper(), seed["seed_name"])
        empty_pages = 0
        for page in range(1, cfg.max_pages_per_seed + 1):
            if len(all_urls) >= target:
                break
            results_url = build_results_url(seed["seed_url"], page)
            try:
                html = fetch_html(session, results_url, cfg)
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", results_url, exc)
                continue
            links = extract_result_links(html, mode=mode)
            new_links = [link for link in links if link not in seen]
            if not new_links:
                empty_pages += 1
                if empty_pages >= 2:
                    break
                continue
            for link in new_links:
                seen.add(link)
                all_urls.append(link)
                if len(all_urls) >= target:
                    break
        logger.info("Collected %s URLs so far for %s", len(all_urls), mode)
    return all_urls[:target]


def extract_residence_jsonld(html: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "lxml")
    for obj in extract_jsonld_objects(soup):
        if obj.get("@type") == "Residence":
            return obj
    return None


def extract_meta_description(soup: BeautifulSoup) -> Optional[str]:
    tag = soup.find("meta", attrs={"name": "description"})
    return clean_text(tag.get("content")) if tag and tag.get("content") else None


def extract_price_from_detail_page(html: str, soup: BeautifulSoup, page_text: str, mode: str) -> Optional[float]:
    candidates = []
    patterns = [r"(R\s*[0-9\s\u00a0\u202f,\.]{3,})"]
    for pattern in patterns:
        m = re.search(pattern, page_text, flags=re.I | re.M)
        if m:
            candidates.append(m.group(1))
    for attr_name, attr_value in [("property", "product:price:amount"), ("property", "og:price:amount"), ("itemprop", "price"), ("property", "price")]:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            candidates.append(tag.get("content"))
    description = extract_meta_description(soup)
    if description:
        candidates.append(description)
    for candidate in candidates:
        amount = parse_money(candidate)
        if amount is None:
            continue
        if mode == "sale" and amount >= 100000:
            return amount
        if mode == "rent" and 500 <= amount <= 100000:
            return amount
    return None


def extract_bed_bath_parking(text: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    bed = bath = park = None
    bed_match = re.search(r"(\d+)\s+bed", text, flags=re.I)
    bath_match = re.search(r"(\d+)\s+bath", text, flags=re.I)
    park_match = re.search(r"(\d+)\s+(?:parking|park)", text, flags=re.I)
    if bed_match:
        bed = int(bed_match.group(1))
    if bath_match:
        bath = int(bath_match.group(1))
    if park_match:
        park = int(park_match.group(1))
    return bed, bath, park


def parse_detail_page(html: str, url: str, mode: str, seed_property_type: Optional[str]) -> dict:
    soup = BeautifulSoup(html, "lxml")
    page_text = soup.get_text(" ", strip=True)
    jsonld = extract_residence_jsonld(html) or {}

    title = first_non_null(clean_text(jsonld.get("name")), clean_text(soup.title.get_text(" ", strip=True) if soup.title else None))
    description = first_non_null(clean_text(jsonld.get("description")), extract_meta_description(soup))
    price_value = extract_price_from_detail_page(html, soup, page_text, mode=mode)

    listing_id = None
    m = re.search(r"/(\d+)$", url.rstrip("/"))
    if m:
        listing_id = m.group(1)
    location = clean_text(jsonld.get("address", {}).get("addressLocality") if isinstance(jsonld.get("address"), dict) else None)
    suburb = location
    city = None
    province = None
    if not suburb:
        parts = [clean_text(x.get_text(" ", strip=True)) for x in soup.select("[class*='breadcrumb'] a, nav a")]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            suburb = parts[-1]
            city = parts[-2]
            province = parts[-3] if len(parts) >= 3 else None

    property_type = first_non_null(clean_text(jsonld.get("@type") if jsonld.get("@type") != "Residence" else None), seed_property_type)
    if property_type and property_type.lower() == "residence":
        property_type = seed_property_type

    bed, bath, park = extract_bed_bath_parking(page_text)
    floor_area = parse_float(first_non_null(re.search(r"([0-9,.]+)\s*m²", page_text).group(1) if re.search(r"([0-9,.]+)\s*m²", page_text) else None, None))
    land_area = None
    levies = parse_money(re.search(r"Levies\s*R?\s*([0-9\s,.]+)", page_text, flags=re.I).group(1) if re.search(r"Levies\s*R?\s*([0-9\s,.]+)", page_text, flags=re.I) else None)
    rates = parse_money(re.search(r"Rates(?: and Taxes)?\s*R?\s*([0-9\s,.]+)", page_text, flags=re.I).group(1) if re.search(r"Rates(?: and Taxes)?\s*R?\s*([0-9\s,.]+)", page_text, flags=re.I) else None)

    slug_parts = [p for p in url.split("privateproperty.co.za/")[-1].split("/") if p]
    row = {
        "source_site": "privateproperty",
        "listing_id": listing_id,
        "title": title,
        "suburb": suburb,
        "city": city,
        "province": province,
        "property_type": seed_property_type if seed_property_type else property_type,
        "bedrooms": bed,
        "bathrooms": bath,
        "parking_spaces": park,
        "garage": park,
        "floor_area_sqm": floor_area,
        "land_area_sqm": land_area,
        "levies": levies,
        "rates_taxes": rates,
        "description": description,
        "listing_date": None,
        "scraped_timestamp": pd.Timestamp.utcnow().isoformat(),
        "pp_transaction_slug": slug_parts[0] if len(slug_parts) > 0 else None,
        "pp_province_slug": slug_parts[1] if len(slug_parts) > 1 else None,
        "pp_metro_slug": slug_parts[2] if len(slug_parts) > 2 else None,
        "pp_city_slug": slug_parts[3] if len(slug_parts) > 3 else None,
        "pp_suburb_slug": slug_parts[4] if len(slug_parts) > 4 else None,
    }
    if mode == "sale":
        row["listing_url"] = url
        row["purchase_price"] = price_value
    else:
        row["rental_url"] = url
        row["monthly_rent"] = price_value
        row["furnished_flag"] = bool(re.search(r"furnished", page_text, flags=re.I))
        availability = re.search(r"Available\s+([A-Za-z0-9 ,]+)", page_text, flags=re.I)
        row["availability_text"] = clean_text(availability.group(1)) if availability else None
    return row


def coerce_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({"": np.nan, "None": np.nan, "none": np.nan, "null": np.nan, "Null": np.nan}) if not df.empty else df.copy()


def dedupe_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    url_col = "listing_url" if mode == "sale" else "rental_url"
    subset = [c for c in ["listing_id", url_col] if c in df.columns]
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def standardize_location_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    for col in ["suburb", "city", "province"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: title_case_slug(x) if pd.notna(x) else x)
    return df


def validate_numeric_ranges(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    for col in ["bedrooms", "bathrooms", "parking_spaces", "garage", "floor_area_sqm", "land_area_sqm", "levies", "rates_taxes"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if mode == "sale" and "purchase_price" in work.columns:
        work["purchase_price"] = pd.to_numeric(work["purchase_price"], errors="coerce")
        work = work[(work["purchase_price"].isna()) | (work["purchase_price"] > 0)]
    if mode == "rent" and "monthly_rent" in work.columns:
        work["monthly_rent"] = pd.to_numeric(work["monthly_rent"], errors="coerce")
        work = work[(work["monthly_rent"].isna()) | ((work["monthly_rent"] >= 500) & (work["monthly_rent"] <= 100000))]
    return work


def finalize_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = LISTINGS_COLS if mode == "sale" else RENTALS_COLS
    out = df.copy()
    out = coerce_nulls(out)
    out = dedupe_frame(out, mode=mode)
    out = standardize_location_columns(out)
    out = validate_numeric_ranges(out, mode=mode)
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols].reset_index(drop=True)


def scrape_mode(mode: str, cfg: ScraperConfig) -> pd.DataFrame:
    target = cfg.max_sale_records if mode == "sale" else cfg.max_rental_records
    session = build_session(cfg.user_agent)
    detail_urls = collect_detail_urls_for_mode(mode=mode, cfg=cfg)
    logger.info("Collected %s detail URLs for mode=%s", len(detail_urls), mode)
    rows = []
    for idx, url in enumerate(detail_urls[:target], start=1):
        seed_property_type = None
        if "apartments" in url:
            seed_property_type = "Apartment"
        elif "houses" in url:
            seed_property_type = "House"
        try:
            html = fetch_html(session, url, cfg)
            rows.append(parse_detail_page(html=html, url=url, mode=mode, seed_property_type=seed_property_type))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", url, exc)
        if idx % cfg.detail_pause_every == 0:
            time.sleep(cfg.long_pause_seconds)
    return finalize_frame(pd.DataFrame(rows), mode=mode)


def build_rental_benchmarks(df_rentals: pd.DataFrame) -> pd.DataFrame:
    if df_rentals.empty or "monthly_rent" not in df_rentals.columns:
        return pd.DataFrame(columns=["suburb", "avg_rent", "median_rent", "listings"])
    work = df_rentals.copy()
    work["monthly_rent"] = pd.to_numeric(work["monthly_rent"], errors="coerce")
    work["suburb"] = work["suburb"].astype("string").str.strip()
    out = work.dropna(subset=["suburb", "monthly_rent"]).groupby("suburb", dropna=False).agg(avg_rent=("monthly_rent", "mean"), median_rent=("monthly_rent", "median"), listings=("monthly_rent", "size")).reset_index()
    out[["avg_rent", "median_rent"]] = out[["avg_rent", "median_rent"]].round(2)
    return out.sort_values(["listings", "avg_rent"], ascending=[False, False])


def build_market_signals(df_listings: pd.DataFrame, df_rentals: pd.DataFrame) -> pd.DataFrame:
    sales_summary = pd.DataFrame(columns=["suburb", "avg_purchase_price", "median_purchase_price", "sale_listings"])
    rent_summary = pd.DataFrame(columns=["suburb", "avg_monthly_rent", "median_monthly_rent", "rental_listings"])
    if not df_listings.empty:
        sales = df_listings.copy()
        sales["purchase_price"] = pd.to_numeric(sales["purchase_price"], errors="coerce")
        sales_summary = sales.dropna(subset=["suburb", "purchase_price"]).groupby("suburb").agg(avg_purchase_price=("purchase_price", "mean"), median_purchase_price=("purchase_price", "median"), sale_listings=("purchase_price", "size")).reset_index()
    if not df_rentals.empty:
        rents = df_rentals.copy()
        rents["monthly_rent"] = pd.to_numeric(rents["monthly_rent"], errors="coerce")
        rent_summary = rents.dropna(subset=["suburb", "monthly_rent"]).groupby("suburb").agg(avg_monthly_rent=("monthly_rent", "mean"), median_monthly_rent=("monthly_rent", "median"), rental_listings=("monthly_rent", "size")).reset_index()
    out = sales_summary.merge(rent_summary, on="suburb", how="outer")
    if {"avg_purchase_price", "avg_monthly_rent"}.issubset(out.columns):
        out["gross_yield_pct_est"] = ((out["avg_monthly_rent"] * 12) / out["avg_purchase_price"] * 100).round(2)
    return out


def save_outputs(df_listings: pd.DataFrame, df_rentals: pd.DataFrame, sales_path: Path, rentals_path: Path) -> tuple[Path, Path]:
    sales_path.parent.mkdir(parents=True, exist_ok=True)
    rentals_path.parent.mkdir(parents=True, exist_ok=True)
    if not df_listings.empty:
        df_listings.to_csv(sales_path, index=False)
        logger.info("Saved sales listings to %s", sales_path)
    if not df_rentals.empty:
        df_rentals.to_csv(rentals_path, index=False)
        logger.info("Saved rentals to %s", rentals_path)
    benchmarks = build_rental_benchmarks(df_rentals)
    market_signals = build_market_signals(df_listings, df_rentals)
    area_dir = sales_path.parents[1] / "area_data"
    area_dir.mkdir(parents=True, exist_ok=True)
    benchmarks.to_csv(area_dir / "privateproperty_rental_benchmarks_by_suburb.csv", index=False)
    market_signals.to_csv(area_dir / "privateproperty_market_signals_by_suburb.csv", index=False)
    return sales_path, rentals_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Private Property scraping pipeline")
    parser.add_argument("--max-sale-records", type=int, default=300)
    parser.add_argument("--max-rental-records", type=int, default=300)
    parser.add_argument("--max-pages-per-seed", type=int, default=20)
    parser.add_argument("--sales-output", default="data/raw/listings/privateproperty_sales_final.csv")
    parser.add_argument("--rentals-output", default="data/raw/rentals/privateproperty_rentals_final.csv")
    args = parser.parse_args()

    cfg = ScraperConfig(max_sale_records=args.max_sale_records, max_rental_records=args.max_rental_records, max_pages_per_seed=args.max_pages_per_seed)
    sales = scrape_mode("sale", cfg)
    rentals = scrape_mode("rent", cfg)
    save_outputs(sales, rentals, Path(args.sales_output), Path(args.rentals_output))
    logger.info("Scraping complete. Sales rows=%s Rentals rows=%s", len(sales), len(rentals))


if __name__ == "__main__":
    main()
