import xarray as xr
import numpy as np
import glob
import os
from tqdm import tqdm
import argparse
import calendar
from datetime import datetime


# ------------------------------------ Setup ------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("Year", type=int, help="The year to process data from")
parser.add_argument("Month", type=int, help="The month to process data from (1-12)")
args = parser.parse_args()

YEAR = args.Year
MONTH = args.Month

GRIB_FILE_DIR = os.path.join("grib_files", YEAR)
TARGET_SHAPE = (480,480)
# NE US slice
SLICE_LAT = slice(750, 750+1250)
SLICE_LON = slice(6200 - 1250, 6200)
# SW US slice
#SLICE_LAT = slice(2450-1250, 2450)
#SLICE_LON = slice(1000, 1000+1250)

MAIN_OUTPUT_DIR = "NE_480/"
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

OUTPUT_FILENAME = os.path.join(MAIN_OUTPUT_DIR, f"{YEAR}{MONTH:02d}.nc")
DAY_DIR = os.path.join(MAIN_OUTPUT_DIR, f"{YEAR}{MONTH:02d}")
os.makedirs(DAY_DIR, exist_ok=True)


_, numDays = calendar.monthrange(YEAR, MONTH)
print(f"Preparing to process all files for {YEAR}-{MONTH:02d}.")


# ------------------------------- Processing Loop ------------------------------- #

for day in range(1, numDays+1):
    fileSavePath = os.path.join(DAY_DIR, f"{day:02d}.nc")
    if os.path.exists(fileSavePath):
        continue
    lstGribFiles = sorted(glob.glob(os.path.join(GRIB_FILE_DIR,f"*{YEAR}{MONTH:02d}{day:02d}*.grib2")))
    lstRadarFrames = []
    for filePath in tqdm(lstGribFiles, desc=f"Day {day:02d} files"):
        try:
            ds = xr.open_dataset(filePath, engine="cfgrib", decode_timedelta=False)
            dataPrec = ds[list(ds.data_vars)[0]]
            dataPrec = dataPrec.isel(latitude=SLICE_LAT, longitude=SLICE_LON)
            # Thresholding
            dataPrec = dataPrec.where(dataPrec>=0, 0)
            # Log Scale
            dataLogPrec = np.log(dataPrec + 1e-4)
            # Interpolating
            coordsLatResized = np.linspace(dataLogPrec.latitude.min().values,
                                           dataLogPrec.latitude.max().values,
                                           TARGET_SHAPE[0])
            coordsLonResized = np.linspace(dataLogPrec.longitude.min().values,
                                           dataLogPrec.longitude.max().values,
                                           TARGET_SHAPE[1])
            dataLogPrecResized = dataLogPrec.interp(latitude=coordsLatResized, longitude=coordsLonResized)
            lstRadarFrames.append(dataLogPrecResized)
        except EOFError:
            tqdm.write(f"WARNING: Skipping corrupted file: {filePath}")
            continue
    ds = xr.Dataset({"log_precipitation": xr.concat(lstRadarFrames, dim="time")})
    ds.to_netcdf(fileSavePath, encoding={"log_precipitation": {"zlib": True, "complevel": 5}})


# ----------------------------- File Concatenation ----------------------------- #
print(f"\nDaily processing complete. Combining all files from '{DAY_DIR}'...")
filesDay = sorted(glob.glob(os.path.join(DAY_DIR, '*.nc')))
if filesDay:
    combined_ds = xr.open_mfdataset(filesDay, combine='nested', concat_dim='time')
    combined_ds.to_netcdf(
        OUTPUT_FILENAME,
        encoding={'log_precipitation': {'zlib': True, 'complevel': 5}}
    )
    print(f"Final combined file saved to '{OUTPUT_FILENAME}'")
else:
    print("No daily files were created.")