import xarray as xr
import os

def CombineFiles(lstFile, fileName, varName='log_precipitation'):
    """
    Combines multiple .nc files to a single .nc file.

    Parameters:
        lstFile (list): list of .nc files to combine.
        fileName (str): Name of output file.
        varName (str): Name of data variable.

    Returns:
        None
    """
    print(f"\n--- Combining {len(lstFile)} files for: {fileName} ---")
    try:
        with xr.open_mfdataset(lstFile, combine='nested', concat_dim='time') as combined_ds:
            print("Combining and saving...")
            encoding = {varName: {'zlib': True, 'complevel': 5}}
            combined_ds.to_netcdf(fileName, encoding=encoding)
        print(f"Successfully created '{fileName}'")
    except Exception as e:
        print(f"An error occurred while creating '{fileName}': {e}")

# ------------------------------------ Setup ------------------------------------ #

FILES_DIR = "NE_480/"

# Training Set
lstTrain = ["202301.nc", "202303.nc", "202305.nc",
            "202307.nc", "202309.nc", "202311.nc"]
lstTrain = [os.path.join(FILES_DIR, f) for f in lstTrain]

# Validation Set
lstVal = ["202401.nc", "202407.nc"]
lstVal = [os.path.join(FILES_DIR, f) for f in lstVal]

# Test Set
lstTest = ["202501.nc", "202507.nc"]
lstTest = [os.path.join(FILES_DIR, f) for f in lstTest]


# ---------------------------------- Combine ----------------------------------- #
CombineFiles(lstTrain, os.path.join(FILES_DIR, "train_dataset_NE_480.nc"))

# Create the Validation Set file
CombineFiles(lstVal, os.path.join(FILES_DIR, "validation_dataset_NE_480.nc"))

# Create the Test Set file
CombineFiles(lstTest, os.path.join(FILES_DIR, "test_dataset_NE_480.nc"))
print("\nAll dataset combination tasks are complete.")
