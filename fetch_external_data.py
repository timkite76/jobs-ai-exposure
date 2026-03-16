"""
Fetch external data from BLS API and O*NET for enhanced analysis.

BLS Public API: No key required (25 series/request, 10yr max).
O*NET API: Requires free key from https://services.onetcenter.org/developer/signup

Usage:
    python3 fetch_external_data.py                    # BLS only (no key needed)
    python3 fetch_external_data.py --onet-key YOUR_KEY  # BLS + O*NET

Environment variables:
    BLS_API_KEY     - Optional, increases rate limit
    ONET_USERNAME   - O*NET Web Services username (from signup)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
ONET_API_URL = "https://services.onetcenter.org/ws/"

# BLS CES series: industry-level employment (thousands), seasonally adjusted
# These map roughly to occupation categories in our data
BLS_INDUSTRY_SERIES = {
    "CES0000000001": "Total Nonfarm",
    "CES0500000001": "Total Private",
    "CES1000000001": "Mining & Logging",
    "CES2000000001": "Construction",
    "CES3000000001": "Manufacturing",
    "CES4000000001": "Trade, Transport & Utilities",
    "CES4142000001": "Retail Trade",
    "CES5000000001": "Information",
    "CES5500000001": "Financial Activities",
    "CES6000000001": "Prof & Business Services",
    "CES6054000001": "Computer Systems Design",
    "CES6500000001": "Education & Health Services",
    "CES6562000001": "Health Care",
    "CES7000000001": "Leisure & Hospitality",
    "CES7072000001": "Food Services",
    "CES8000000001": "Other Services",
    "CES9000000001": "Government",
    "CES9091000001": "Federal Government",
    "CES9092000001": "State Government",
    "CES9093000001": "Local Government",
}

# Map our occupation categories to BLS industry codes for cross-referencing
CATEGORY_TO_INDUSTRY = {
    "architecture-and-engineering": ["CES6000000001"],
    "arts-and-design": ["CES5000000001", "CES7000000001"],
    "business-and-financial": ["CES5500000001", "CES6000000001"],
    "community-and-social-service": ["CES6500000001"],
    "computer-and-information-technology": ["CES5000000001", "CES6054000001"],
    "construction-and-extraction": ["CES2000000001"],
    "education-training-and-library": ["CES6500000001"],
    "entertainment-and-sports": ["CES7000000001"],
    "farming-fishing-and-forestry": ["CES1000000001"],
    "food-preparation-and-serving": ["CES7072000001"],
    "healthcare": ["CES6562000001"],
    "installation-maintenance-and-repair": ["CES2000000001", "CES3000000001"],
    "legal": ["CES6000000001"],
    "life-physical-and-social-science": ["CES6000000001"],
    "management": ["CES6000000001"],
    "math": ["CES5000000001", "CES6000000001"],
    "media-and-communication": ["CES5000000001"],
    "military": ["CES9091000001"],
    "office-and-administrative-support": ["CES6000000001", "CES5500000001"],
    "personal-care-and-service": ["CES7000000001"],
    "production": ["CES3000000001"],
    "protective-service": ["CES9093000001"],
    "sales": ["CES4142000001"],
    "transportation-and-material-moving": ["CES4000000001"],
}


def fetch_bls_trends(api_key: str | None = None) -> dict:
    """Fetch 10 years of monthly employment data from BLS public API."""
    series_ids = list(BLS_INDUSTRY_SERIES.keys())
    headers = {"Content-Type": "application/json"}

    all_results = {}

    # BLS limits to 25 series per request
    for i in range(0, len(series_ids), 25):
        batch = series_ids[i:i + 25]
        payload = {
            "seriesid": batch,
            "startyear": "2015",
            "endyear": "2024",
        }
        if api_key:
            payload["registrationkey"] = api_key

        print(f"  Fetching BLS batch {i // 25 + 1} ({len(batch)} series)...")
        response = httpx.post(BLS_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "REQUEST_SUCCEEDED":
            print(f"  BLS API error: {data.get('message', 'unknown')}")
            continue

        for series in data["Results"]["series"]:
            sid = series["seriesID"]
            points = []
            for d in series["data"]:
                if d["period"].startswith("M"):
                    month = int(d["period"][1:])
                    points.append({
                        "year": int(d["year"]),
                        "month": month,
                        "value": float(d["value"]),
                    })
            points.sort(key=lambda p: (p["year"], p["month"]))
            all_results[sid] = {
                "name": BLS_INDUSTRY_SERIES.get(sid, sid),
                "data": points,
            }

        if i + 25 < len(series_ids):
            time.sleep(1)

    return {
        "series": all_results,
        "category_mapping": CATEGORY_TO_INDUSTRY,
    }


def fetch_onet_skills(username: str) -> dict:
    """Fetch skill requirements for all occupations from O*NET API."""
    auth = (username, "")  # O*NET uses username as basic auth, no password
    headers = {"Accept": "application/json"}

    # First get the list of occupations
    print("  Fetching O*NET occupation list...")
    resp = httpx.get(
        f"{ONET_API_URL}online/occupations/",
        auth=auth, headers=headers, timeout=30,
        params={"start": 1, "end": 50},
    )
    if resp.status_code == 401:
        print("  O*NET auth failed. Check your username/key.")
        return {}
    resp.raise_for_status()

    # We need SOC-level data. Our occupations use BLS SOC codes.
    # Load our data to get SOC codes
    data_path = Path(__file__).parent / "occupations.csv"
    if not data_path.exists():
        print("  occupations.csv not found, skipping O*NET")
        return {}

    import csv
    with open(data_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = {}
    soc_codes = set()
    for row in rows:
        soc = row.get("soc_code", "").strip()
        if soc:
            soc_codes.add(soc)

    print(f"  Fetching skills for {len(soc_codes)} SOC codes...")
    for i, soc in enumerate(sorted(soc_codes)):
        # O*NET uses extended SOC format: XX-XXXX.00
        onet_soc = f"{soc}.00"
        try:
            resp = httpx.get(
                f"{ONET_API_URL}online/occupations/{onet_soc}/summary/skills",
                auth=auth, headers=headers, timeout=15,
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            skills_data = resp.json()
            skills = []
            for s in skills_data.get("element", []):
                skills.append({
                    "name": s.get("name", ""),
                    "id": s.get("id", ""),
                    "score": float(s.get("score", {}).get("value", 0)),
                })
            results[soc] = {"skills": skills}
        except Exception as e:
            print(f"    Error for {soc}: {e}")

        if (i + 1) % 20 == 0:
            print(f"    Progress: {i + 1}/{len(soc_codes)}")
            time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch external data for jobs analysis")
    parser.add_argument("--bls-key", default=os.getenv("BLS_API_KEY"))
    parser.add_argument("--onet-key", default=os.getenv("ONET_USERNAME"))
    args = parser.parse_args()

    out_dir = Path(__file__).parent

    # BLS historical trends (always runs - no key needed)
    print("Fetching BLS historical employment trends...")
    try:
        bls_data = fetch_bls_trends(args.bls_key)
        with open(out_dir / "bls_trends.json", "w") as f:
            json.dump(bls_data, f)
        print(f"  Saved {len(bls_data['series'])} series to bls_trends.json")
    except Exception as e:
        print(f"  BLS fetch failed: {e}")

    # O*NET skills (only if key provided)
    if args.onet_key:
        print("Fetching O*NET skill requirements...")
        try:
            onet_data = fetch_onet_skills(args.onet_key)
            with open(out_dir / "onet_skills.json", "w") as f:
                json.dump(onet_data, f)
            print(f"  Saved skills for {len(onet_data)} occupations to onet_skills.json")
        except Exception as e:
            print(f"  O*NET fetch failed: {e}")
    else:
        print("Skipping O*NET (no key). Sign up free at https://services.onetcenter.org/developer/signup")


if __name__ == "__main__":
    main()
