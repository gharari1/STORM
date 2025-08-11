import os
import sys
import requests
import gzip
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.append(os.path.join('..', 'models'))

from STORM import STORM

# ---------------------------------- Config ------------------------------------ #
YEAR = 2022
MONTH = 2
DAY = 4
HOUR = 17
MINUTE = 40
AREA = 'NE' # 'NE' or 'SW'
WEIGHTS_PATH = "lr0.0001_d0.1.weights.h5"

INPUT_SHAPE = (4, 480, 480, 1)
DROPOUT_RATE = 0.1

if AREA == 'NE':
    LAT_SLICE = slice(750, 750+1250)
    LON_SLICE = slice(6200 - 1250, 6200)
else: # SW
    LAT_SLICE = slice(2450-1250, 2450)
    LON_SLICE = slice(1000, 1000+1250)

TARGET_SHAPE = (480,480)
GRIB_FILE_DIR = "./grib_files/"
WEIGHTS_PATH = os.path.join("..","saved_weights", WEIGHTS_PATH)

# ---------------------------- Helper Functions ------------------------------ #
def download_and_unzip(dt, dirBase):
    """
    Downloads and unzips a single GRIB file for a given datetime.

    Parameters:
        dt (datetime.datetime): the date and time of file to download.
        dirBase (str): The directory to save files in.
    """
    os.makedirs(dirBase, exist_ok=True)

    strDate = dt.strftime('%Y%m%d')
    strTime = dt.strftime('%H%M%S')
    filename = f"MRMS_PrecipRate_00.00_{strDate}-{strTime}.grib2.gz"

    uncompressed_path = os.path.join(dirBase, filename.replace('.gz', ''))

    if os.path.exists(uncompressed_path):
        return uncompressed_path

    try:
        year, month, day = dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d')
        s3_url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/PrecipRate_00.00/{year+month+day}/{filename}"

        print(f"Downloading {filename}...")
        response = requests.get(s3_url)
        response.raise_for_status()

        uncompressed_data = gzip.decompress(response.content)
        with open(uncompressed_path, 'wb') as f_out:
            f_out.write(uncompressed_data)
        return uncompressed_path
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download {filename}. Error: {e}")
        return None

def preprocess_frame(arrData):
    """
    Applies preprocessing pipeline.

    Parameters:
        arrData (xarray): xarray to be processed.
    """
    data_sliced = arrData.isel(latitude=LAT_SLICE, longitude=LON_SLICE)
    data_cleaned = data_sliced.where(data_sliced >= 0, 0)
    data_log = np.log(data_cleaned + 1e-4)
    coords_lat = np.linspace(data_log.latitude.min().values, data_log.latitude.max().values, TARGET_SHAPE[0])
    coords_lon = np.linspace(data_log.longitude.min().values, data_log.longitude.max().values, TARGET_SHAPE[1])
    data_resized = data_log.interp(latitude=coords_lat, longitude=coords_lon)
    return data_resized

# ---------------------------- Download Files ------------------------------ #
timeBase = datetime(YEAR, MONTH, DAY, HOUR, MINUTE)
timeInterval = timedelta(minutes=6)
lstDateTimes = [timeBase - (3 - ii) * timeInterval for ii in range(4)] # t-18, t-12, t-6, t
lstDateTimes.append(timeBase + timeInterval)                           # t+6

print("Downloading files")
pathFiles = [download_and_unzip(dt, GRIB_FILE_DIR) for dt in lstDateTimes]
if not all(pathFiles):
    print("\nCould not download all necessary files. Exiting.")
    exit()

pathInput = pathFiles[:-1]
pathGroundTruth = pathFiles[-1]

# ---------------------------- Process Files ------------------------------ #
print("\nPreprocessing data")
processedInput = []
slicedInputXR = []
for path in pathInput:
    with xr.open_dataset(path, engine="cfgrib", decode_timedelta=False) as ds:
        frame = ds[list(ds.data_vars)[0]]
        processedFrame = preprocess_frame(frame)
        processedInput.append(processedFrame.values)
        slicedInputXR.append(np.maximum(0, np.exp(processedFrame) - 1e-4))

tensorInput = np.stack(processedInput, axis=0)
tensorInput = np.expand_dims(tensorInput, axis=[0, -1])

with xr.open_dataset(pathGroundTruth, engine="cfgrib", decode_timedelta=False) as ds:
    frame = ds[list(ds.data_vars)[0]]
    groundTruth = preprocess_frame(frame)

# ------------------------------- Predict --------------------------------- #
print("\nLoading model")
model = STORM(input_shape=INPUT_SHAPE, mode="regression", dropout=DROPOUT_RATE)
model.load_weights(WEIGHTS_PATH)
print(f"Successfully loaded weights from {WEIGHTS_PATH}")

logPred = model.predict(tensorInput)
logPred = np.squeeze(logPred)

mmhrPred = np.maximum(0, np.exp(logPred) - 1e-4)
mmhrGroundTruth = np.maximum(0, np.exp(groundTruth.values) - 1e-4)

# ------------------------------- Visualize --------------------------------- #
plt.rcParams['font.family'] = 'serif'
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
vmin = 1.0
vmax = 10
cmap = plt.get_cmap('viridis')
cmap.set_under('none')

# Plot input frames on the first row
for ii in range(4):
    slicedInputXR[ii].plot(ax=axes[0, ii], cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    axes[0, ii].set_title(f"Input t-{(3 - ii) * 6} min")

xrGroundTruth = xr.DataArray(data=mmhrGroundTruth, coords=processedFrame.coords)
xrGroundTruth.plot(ax=axes[1, 1], cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
axes[1, 1].set_title("Ground Truth (t+6 min)")

xrPred = xr.DataArray(data=mmhrPred, coords=processedFrame.coords)
im = xrPred.plot(ax=axes[1, 2], cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
axes[1, 2].set_title("STORM Prediction (t+6 min)")

axes[1, 0].axis('off')
axes[1, 3].axis('off')

fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, extend='both')
cbar.set_label('Precipitation Rate (mm/hr)', size=12)

plt.savefig(f"evaluation_{AREA}_{timeBase.strftime('%Y%m%d_%H%M')}.png")
plt.show()