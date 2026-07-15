import os
import re
import csv
from datetime import datetime

INPUT_DIR  = "metar_raw"
OUTPUT_CSV = "vabb_metar_clean.csv"

# ── HELPER FUNCTIONS ─────────────────────────────────────

def parse_timestamp(raw_ts):
    # raw_ts looks like: 202412010000
    # means: year=2024, month=12, day=01, hour=00, min=00
    try:
        return datetime.strptime(raw_ts, "%Y%m%d%H%M")
    except:
        return None

def parse_wind(metar_str):
    # Matches: 28010KT or 28010G25KT or VRB05KT or 00000KT
    gust_match = re.search(r'(\d{3}|VRB)(\d{2,3})G(\d{2,3})KT', metar_str)
    wind_match  = re.search(r'(\d{3}|VRB)(\d{2,3})KT', metar_str)

    if gust_match:
        direction = None if gust_match.group(1) == 'VRB' else int(gust_match.group(1))
        speed     = int(gust_match.group(2))
        gust      = int(gust_match.group(3))
    elif wind_match:
        direction = None if wind_match.group(1) == 'VRB' else int(wind_match.group(1))
        speed     = int(wind_match.group(2))
        gust      = speed   # no gust reported → fill with speed
    else:
        direction, speed, gust = None, None, None

    return direction, speed, gust

def parse_visibility(metar_str):
    # Matches: 4000 or 9999 or CAVOK
    if 'CAVOK' in metar_str:
        return 10000
    vis_match = re.search(r'\s(\d{4})\s', metar_str)
    if vis_match:
        val = int(vis_match.group(1))
        return 10000 if val == 9999 else val
    return None

def parse_temp_dewpoint(metar_str):
    # Matches: 32/25 or M02/M05 (M = minus)
    match = re.search(r'(M?\d{2})/(M?\d{2})', metar_str)
    if match:
        temp = match.group(1).replace('M', '-')
        dewp = match.group(2).replace('M', '-')
        return int(temp), int(dewp)
    return None, None

def parse_pressure(metar_str):
    # Matches: Q1008
    match = re.search(r'Q(\d{4})', metar_str)
    if match:
        return int(match.group(1))
    return None

# ── MAIN PARSER ──────────────────────────────────────────

rows = []

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(INPUT_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Only process actual METAR lines
            if "METAR VABB" not in line:
                continue

            # Skip corrupt/incomplete lines
            parts = line.split()
            if len(parts) < 5:
                continue

            raw_ts   = parts[0]           # e.g. 202412010000
            metar_body = " ".join(parts)  # full line for regex parsing

            timestamp            = parse_timestamp(raw_ts)
            wind_dir, speed, gust = parse_wind(metar_body)
            visibility           = parse_visibility(metar_body)
            temp, dewpoint       = parse_temp_dewpoint(metar_body)
            pressure             = parse_pressure(metar_body)

            if timestamp is None:
                continue

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

# Sort by time (important — files may not be perfectly ordered)
rows.sort(key=lambda x: x["timestamp"])

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["timestamp","wind_dir","wind_speed","gust",
                  "visibility","temp","dewpoint","pressure"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Done. {len(rows)} rows written to {OUTPUT_CSV}")