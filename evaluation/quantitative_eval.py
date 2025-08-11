import os
import sys
import numpy as np
import xarray as xr
import tensorflow as tf
from tqdm import tqdm
import random

sys.path.append("..//training//")
sys.path.append("..//models//")

from data_loader import CreateDataset
from STORM import STORM

# ---------------------------------- Config ------------------------------------ #
INPUT_SHAPE = (4,480,480,1)
MODEL_WEIGHTS = ""
TEST_DATA = ""
SEED = 0
DROPOUT_RATE = 0.5

# Seeding
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------- Setup ------------------------------------ #
print(f"Loading and cleaning {TEST_DATA} into memory")
with xr.open_dataset(os.path.join("..","datasets",TEST_DATA)) as ds:
    arrTestData = ds[list(ds.data_vars)[0]]
    dataTest = arrTestData.values
    tsTest = arrTestData['time'].values

datasetTest, stepsTest = CreateDataset(dataTest, 
                                       tsTest, 
                                       batch_siz=1, 
                                       input_shape=INPUT_SHAPE, 
                                       shuffle=False)

# Build, load weights and compile and the model
print("\nBuilding and Compiling Model")
model = STORM(input_shape=INPUT_SHAPE, mode="regression", dropout=DROPOUT_RATE)
model.load_weights(os.path.join("..","saved_weights",MODEL_WEIGHTS))
print(f"Successfully loaded weights from: {MODEL_WEIGHTS}")

model.compile(loss=tf.keras.losses.LogCosh(),
              metrics=['mean_absolute_error'])


# ---------------------------- Run Evaluation ------------------------------ #
print("\nStarting Evaluation on Test Set")

lstPred_mmhr = []
lstGroundTruth_mmhr = []

# Loop through the entire test set
for batchX, batch_yLog in tqdm(datasetTest, total=stepsTest, desc="Evaluating"):
    # Predict
    logPred = model.predict_on_batch(batchX)
    # Inverse transform to mm/hr
    batch_y_mmhr = np.maximum(0, np.exp(np.squeeze(batch_yLog, axis=-1)) - 1e-4)
    predictions_mmhr = np.maximum(0, np.exp(np.squeeze(logPred, axis=-1)) - 1e-4)
    
    lstPred_mmhr.append(predictions_mmhr)
    lstGroundTruth_mmhr.append(batch_y_mmhr)

# Concatenate
lstPred_mmhr = np.concatenate(lstPred_mmhr)
lstGroundTruth_mmhr = np.concatenate(lstGroundTruth_mmhr)

# Standard regression metrics
mae = np.mean(np.abs(lstPred_mmhr - lstGroundTruth_mmhr))
rmse = np.sqrt(np.mean((lstPred_mmhr - lstGroundTruth_mmhr) ** 2))

# Conditional regression metrics
thresholdPrecipitation = 0.1
maskPrecipitation = lstGroundTruth_mmhr > thresholdPrecipitation
condPred = lstPred_mmhr[maskPrecipitation]
condGroundTruth = lstGroundTruth_mmhr[maskPrecipitation]

condMAE = np.mean(np.abs(condPred - condGroundTruth))
condRMSE = np.sqrt(np.mean((condPred - condGroundTruth) ** 2))

# Skill score
thresholdPrecipitationSkill = 1.0
groundTruthPrec = lstGroundTruth_mmhr >= thresholdPrecipitationSkill
predPrec = lstPred_mmhr >= thresholdPrecipitationSkill

hits = np.sum(groundTruthPrec & predPrec)
misses = np.sum(groundTruthPrec & ~predPrec)
false_alarms = np.sum(~groundTruthPrec & predPrec)

csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
pod = hits / (hits + misses) if (hits + misses) > 0 else 0


print("\n" + "-"*50)
print(f"--- Final Test Set Results for {TEST_DATA} ---")
print("\n--- Overall Regression Metrics (All Pixels) ---")
print(f"Mean Absolute Error (MAE): {mae:.4f} mm/hr")
print(f"Root Mean Square Error (RMSE): {rmse:.4f} mm/hr")
print("\n--- Conditional Regression Metrics (Rain Pixels Only) ---")
print(f"MAE on Rain Pixels (> {thresholdPrecipitation} mm/hr): {condMAE:.4f} mm/hr")
print(f"RMSE on Rain Pixels (> {thresholdPrecipitation} mm/hr): {condRMSE:.4f} mm/hr")
print(f"\n--- Skill Scores (Threshold = {thresholdPrecipitationSkill} mm/hr) ---")
print(f"Critical Success Index (CSI): {csi:.4f}")
print(f"Probability of Detection (POD): {pod:.4f}")
print("-"*50)