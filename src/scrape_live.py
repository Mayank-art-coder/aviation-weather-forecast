"""
src/scrape_live.py
Scrapes latest 3 days of METAR from ogimet.com for VABB.
Used by Airflow DAG every 30 minutes to get fresh observations.
Reuses your existing selenium + regex parsing logic exactly.
"""
import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger import get_logger

logger = get_logger(__name__)

STATION    = "vabb"
FORM_URL   = "https://www.ogimet.com/metars.phtml.en"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── SAME PARSERS AS parsetxt_to_csv.py ────────────────
def parse_wind(metar_str):
    gust_match = re.search(r'(\d{3}|VRB)(\d{2,3})G(\d{2,3})KT', metar_str)
    wind_match  = re.search(r'(\d{3}|VRB)(\d{2,3})KT', metar_str)
    if gust_match:
        direction = None if gust_match.group(1) == 'VRB' else int(gust_match.group(1))
        speed     = int(gust_match.group(2))
        gust      = int(gust_match.group(3))
    elif wind_match:
        direction = None if wind_match.group(1) == 'VRB' else int(wind_match.group(1))
        speed     = int(wind_match.group(2))
        gust      = speed
    else:
        direction, speed, gust = None, None, None
    return direction, speed, gust

def parse_visibility(metar_str):
    if 'CAVOK' in metar_str:
        return 10000
    vis_match = re.search(r'\s(\d{4})\s', metar_str)
    if vis_match:
        val = int(vis_match.group(1))
        return 10000 if val == 9999 else val
    return None

def parse_temp_dewpoint(metar_str):
    match = re.search(r'(M?\d{2})/(M?\d{2})', metar_str)
    if match:
        temp = match.group(1).replace('M', '-')
        dewp = match.group(2).replace('M', '-')
        return int(temp), int(dewp)
    return None, None

def parse_pressure(metar_str):
    match = re.search(r'Q(\d{4})', metar_str)
    return int(match.group(1)) if match else None

def parse_timestamp(raw_ts):
    try:
        return datetime.strptime(raw_ts, "%Y%m%d%H%M")
    except:
        return None

def parse_metar_lines(raw_text: str) -> list:
    """Parse raw METAR text into list of dicts — same logic as parsetxt_to_csv.py"""
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if "METAR VABB" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        raw_ts    = parts[0]
        metar_body = " ".join(parts)
        timestamp  = parse_timestamp(raw_ts)
        if timestamp is None:
            continue
        wind_dir, speed, gust = parse_wind(metar_body)
        visibility             = parse_visibility(metar_body)
        temp, dewpoint         = parse_temp_dewpoint(metar_body)
        pressure               = parse_pressure(metar_body)
        rows.append({
            "timestamp":  timestamp,
            "wind_dir":   wind_dir,
            "wind_speed": speed,
            "gust":       gust,
            "visibility": visibility,
            "temp":       temp,
            "dewpoint":   dewpoint,
            "pressure":   pressure
        })
    return rows


# ── SELENIUM SCRAPER — last N days ────────────────────
def scrape_recent_metars(days_back: int = 3) -> pd.DataFrame:
    """
    Scrapes last `days_back` days of METAR from ogimet.
    Returns raw parsed DataFrame (not yet feature-engineered).
    Uses headless Chrome — no GUI needed on server.
    """
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait, Select
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options

    now   = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)

    logger.info(f"Scraping VABB METARs from {start.date()} to {now.date()}")

    # Headless Chrome — works on Ubuntu server without display
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    wait   = WebDriverWait(driver, 30)

    try:
        driver.get(FORM_URL)
        wait.until(EC.presence_of_element_located((By.NAME, "lugar")))

        # Fill form — same as your batch script
        driver.find_element(By.NAME, "lugar").clear()
        driver.find_element(By.NAME, "lugar").send_keys(STATION)

        Select(driver.find_element(By.NAME, "fmt")).select_by_value("txt")
        Select(driver.find_element(By.NAME, "ord")).select_by_visible_text("Oldest first")
        Select(driver.find_element(By.NAME, "nil")).select_by_visible_text("NIL report excluded")

        # Start date
        Select(driver.find_element(By.NAME, "ano")).select_by_visible_text(str(start.year))
        Select(driver.find_element(By.NAME, "mes")).select_by_visible_text(
            ["","January","February","March","April","May","June",
             "July","August","September","October","November","December"][start.month]
        )
        Select(driver.find_element(By.NAME, "day")).select_by_visible_text(str(start.day).zfill(2))
        Select(driver.find_element(By.NAME, "hora")).select_by_visible_text("00")

        # End date
        Select(driver.find_element(By.NAME, "anof")).select_by_visible_text(str(now.year))
        Select(driver.find_element(By.NAME, "mesf")).select_by_visible_text(
            ["","January","February","March","April","May","June",
             "July","August","September","October","November","December"][now.month]
        )
        Select(driver.find_element(By.NAME, "dayf")).select_by_visible_text(str(now.day).zfill(2))
        Select(driver.find_element(By.NAME, "horaf")).select_by_visible_text("23")

        driver.find_element(By.NAME, "send").click()
        pre = wait.until(EC.presence_of_element_located((By.TAG_NAME, "pre")))
        raw_text = pre.text
        logger.info(f"Scraped {raw_text.count('METAR')} METAR lines")

    finally:
        driver.quit()

    rows = parse_metar_lines(raw_text)
    if not rows:
        raise ValueError("No METAR data parsed from ogimet response")

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Save raw to disk for audit trail
    outfile = OUTPUT_DIR / f"live_metar_{now.strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(outfile, index=False)
    logger.info(f"Saved raw scrape: {outfile} ({len(df)} rows)")

    return df
