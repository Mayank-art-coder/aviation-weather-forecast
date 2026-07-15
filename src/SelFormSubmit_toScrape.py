from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import os

OUTPUT_DIR = "metar_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR = 2014
END_YEAR = 2024

DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def last_day(year, month):
    if month == 2:
        return 29 if is_leap(year) else 28
    return DAYS_IN_MONTH[month]

FORM_URL = "https://www.ogimet.com/metars.phtml.en"

driver = webdriver.Edge()
wait = WebDriverWait(driver, 30)

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):

        filename = f"{OUTPUT_DIR}/VABB_{year}_{month:02d}.txt"

        if os.path.exists(filename):
            print(f"SKIPPING {year}-{month:02d} (already exists)")
            continue

        print(f"Fetching {year}-{month:02d} ...")

        try:
            # Step 1: Go to form page
            driver.get(FORM_URL)

            # Step 2: Wait for form to load
            wait.until(EC.presence_of_element_located((By.NAME, "lugar")))

            # Step 3: Fill station code
            station_field = driver.find_element(By.NAME, "lugar")
            station_field.clear()
            station_field.send_keys("vabb")

            # Step 4: Select format = TXT
            Select(driver.find_element(By.NAME, "fmt")).select_by_value("txt")

            # Step 5: Select order = oldest first (check what value your form uses)
            Select(driver.find_element(By.NAME, "ord")).select_by_visible_text("Oldest first")

            # Step 6: Select NIL excluded
            Select(driver.find_element(By.NAME, "nil")).select_by_visible_text("NIL report excluded")

            # Step 7: Start year/month/day
            Select(driver.find_element(By.NAME, "ano")).select_by_visible_text(str(year))
            Select(driver.find_element(By.NAME, "mes")).select_by_visible_text(
                ["","January","February","March","April","May","June",
                 "July","August","September","October","November","December"][month]
            )
            Select(driver.find_element(By.NAME, "day")).select_by_visible_text("01")
            Select(driver.find_element(By.NAME, "hora")).select_by_visible_text("00")

            # Step 8: End year/month/day
            Select(driver.find_element(By.NAME, "anof")).select_by_visible_text(str(year))
            Select(driver.find_element(By.NAME, "mesf")).select_by_visible_text(
                ["","January","February","March","April","May","June",
                 "July","August","September","October","November","December"][month]
            )
            Select(driver.find_element(By.NAME, "dayf")).select_by_visible_text(
                str(last_day(year, month))
            )
            Select(driver.find_element(By.NAME, "horaf")).select_by_visible_text("23")

            # Step 9: Click send
            driver.find_element(By.NAME, "send").click()

            # Step 10: Wait for result page with <pre> tag
            pre_element = wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "pre"))
            )

            raw_text = pre_element.text

            if "METAR" not in raw_text:
                print(f"WARNING: No METAR data for {year}-{month:02d}")
                with open(filename, "w") as f:
                    f.write("#NO DATA\n")
            else:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(raw_text)
                line_count = raw_text.count("METAR")
                print(f"OK: {line_count} METARs saved → {filename}")

        except Exception as e:
            print(f"ERROR {year}-{month:02d}: {e}")
            with open(f"{OUTPUT_DIR}/ERRORS.log", "a") as log:
                log.write(f"{year}-{month:02d}: {str(e)}\n")

        time.sleep(7)

driver.quit()
print(f"Done. All files in: {OUTPUT_DIR}")