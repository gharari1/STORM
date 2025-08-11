import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import tensorflow as tf
import math

def FindValidIdx(timestamps, timeInterval, numInputFrames=4):
    """
    Scans the time axis to find all indices that can start a valid sequence.

    Parameters:
        timestamps (np.array): array of timestamps.
        timeInterval (np.timedelta64): time interval between frames.
        numInputFrames (int): number of frames to check.

    Returns:
        list: starting indices of valid sequences.
    """
    print("Scanning dataset to find all valid time sequences...")
    lstValidIdx = []
    timestamp_set = set(timestamps)
    for i, start_time in enumerate(tqdm(timestamps, desc="Pre-calculating samples")):
        try:
            required_times = [start_time + (j * timeInterval) for j in range(1, numInputFrames + 1)]
            if all(t in timestamp_set for t in required_times):
                lstValidIdx.append(i)
        except Exception:
            continue
    print(f"Found {len(lstValidIdx)} valid starting points.")
    return lstValidIdx

def CreateDataset(npData, npTimestamps, batch_size, input_shape, shuffle=True):
    """
    Create dataset for STORM model from numpy data.

    Parameters:
        npData (np.array): numpy array with shape [batch_size, height, width, channels].
        npTimestamps (np.array): numpy array timestamps of npData.
        batch_size (int): batch size.
        input_shape (tuple): shape of input frames (batch_size, height, width, channels).

    Returns:
        tf.data.Dataset
        int: number of batches in the dataset
    """
    time_interval = np.timedelta64(6, 'm')
    valid_indices = FindValidIdx(npTimestamps, time_interval, input_shape[0])
    
    dataset = tf.data.Dataset.from_tensor_slices(valid_indices)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(valid_indices))

    def LoadAndProcess(start_idx):
        start_idx = start_idx.numpy()
        input_indices = [start_idx + ii for ii in range(4)]
        output_index = start_idx + 4
        X_frames = npData[input_indices]
        y_frame = npData[output_index]
        X = np.expand_dims(X_frames, axis=-1)
        y = np.expand_dims(y_frame, axis=-1)
        return X.astype(np.float32), y.astype(np.float32)

    def wrapper_LoadAndProcess(idx):
        X, y = tf.py_function(LoadAndProcess, [idx], [tf.float32, tf.float32])
        X.set_shape(input_shape)
        y.set_shape(input_shape[1:])
        return X, y

    dataset = dataset.map(
        wrapper_LoadAndProcess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    numSteps = math.ceil(len(valid_indices) / batch_size)
    
    return dataset, numSteps