import os
import sys
import yaml
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import random

sys.path.append(os.path.join("..","models"))

from data_loader import CreateDataset
from STORM import STORM

# ------------------------ Load Configuration from YAML -------------------------- #
print("--- Loading Configuration from config.yaml ---")
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Convert input_shape from list to tuple
config['input_shape'] = tuple(config['input_shape'])

# Seeding
SEED = config['seed']
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Checkpoint path
dirCheckpoint = os.path.join("checkpoints", wandb.run.id)
os.makedirs(dirCheckpoint, exist_ok=True)
pathCheckpoint = os.path.join(dirCheckpoint, "cp-{epoch:02d}.weights.h5")

# ---------------------------------- Setup ------------------------------------ #
wandb.init(project=config['wandb_project'],
           name=config['wandb_name'],
           config=config)
config = wandb.config

print(f"Loading and cleaning {config.train_data} into memory")
with xr.open_dataset(os.path.join("..","Datasets", config.train_data)) as ds:
    arrTrainData = ds[list(ds.data_vars)[0]]
    dataTrain = arrTrainData.values             # Numpy array in memory
    tsTrain = arrTrainData['time'].values       # Numpy array in memory

print(f"Loading and cleaning {config.val_data} into memory")
with xr.open_dataset(os.path.join("..","Datasets", config.val_data)) as ds:
    arrValData = ds[list(ds.data_vars)[0]]
    dataVal = arrValData.values                 # Numpy array in memory
    tsVal = arrValData['time'].values           # Numpy array in memory

datasetTrain, stepsTrain = CreateDataset(dataTrain,
                                         tsTrain,
                                         config.batch_size,
                                         config.input_shape,
                                         shuffle=True)
datasetVal, stepsVal = CreateDataset(dataVal,
                                     tsVal,
                                     config.batch_size,
                                     config.input_shape,
                                     shuffle=False)

print("\nBuilding and Compiling Model")


model = STORM(input_shape=config.input_shape,
              mode="regression",
              dropout=config.dropout)

print("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss=tf.keras.losses.LogCosh(),
    metrics=['mean_absolute_error']
)

callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
             ModelCheckpoint(filepath=pathCheckpoint, save_weights_only=True, save_freq='epoch', verbose=1),
             WandbMetricsLogger(log_freq="epoch")]

# -------------------------------- Training ----------------------------------- #
print("\nStarting Training")
history = model.fit(datasetTrain,
                    validation_data=datasetVal,
                    epochs=config.epochs,
                    callbacks=callbacks,
                    steps_per_epoch=stepsTrain,
                    validation_steps=stepsVal)


# ---------------------------- Save Final Model ------------------------------ #
print("\nTraining Complete. Saving final model and plotting history.")
model.save('final_model.h5')

artifact = wandb.Artifact('storm-model', type='model')
artifact.add_file('final_model.h5')
wandb.log_artifact(artifact)

wandb.finish()