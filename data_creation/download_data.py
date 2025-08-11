import os
import requests
import gzip
import shutil
import calendar
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse

# ------------------------------------ Setup ------------------------------------ #
BASE_URL = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/PrecipRate_00.00/"
TARGET_MINUTES = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]

parser = argparse.ArgumentParser()
parser.add_argument("Year", type=int, help="The year to download data")
parser.add_argument("Month", type=int, help="The month to download data (1-12)")
args = parser.parse_args()

YEAR = args.Year
MONTH = args.Month

LOCAL_DIRECTORY = f"grib_files/{YEAR}"
os.makedirs(LOCAL_DIRECTORY, exist_ok=True)

_, num_days = calendar.monthrange(YEAR, MONTH)
print(f"Preparing to download all files for {YEAR}-{MONTH:02d} ({num_days} days).")


# -------------------------------- Download Loop -------------------------------- #
total_files = num_days * 24 * len(TARGET_MINUTES)
with tqdm(total=total_files, desc="Overall Progress") as pbar:
    # Loop through each day of the month
    for day in range(1, num_days + 1):
        # Loop through each hour in the day
        for hour in range(24):
            # Loop through the target minutes
            for min in TARGET_MINUTES:
                currTime = datetime(YEAR, MONTH, day, hour, min, 0)
                strDate = currTime.strftime('%Y%m%d')
                strTime = currTime.strftime('%H%M%S')
                fileName = f"MRMS_PrecipRate_00.00_{strDate}-{strTime}.grib2.gz"
                strYear = currTime.strftime('%Y')
                strMonth = currTime.strftime('%m')
                strDay = currTime.strftime('%d')
                fileURL = f"{BASE_URL}{strYear+strMonth+strDay}/{fileName}"
                filePath = os.path.join(LOCAL_DIRECTORY, fileName.replace('.gz', ''))
                # Check if file exists before downloading
                if not os.path.exists(filePath):
                    try:
                        # Download the file
                        response = requests.get(fileURL, timeout=30)
                        response.raise_for_status()
                        # Unzip directly from memory and save
                        uncompressed_data = gzip.decompress(response.content)
                        with open(filePath, 'wb') as f_out:
                            f_out.write(uncompressed_data)
                    except requests.exceptions.RequestException as e:
                        pbar.write(f"\nSkipping {fileName}. Reason: {e}")
                pbar.update(1)


print(f"\nBulk download for {YEAR}-{MONTH:02d} complete.")