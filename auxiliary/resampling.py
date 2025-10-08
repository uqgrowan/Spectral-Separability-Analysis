import os
import glob
import pandas as pd
import numpy as np

def get_rsr_functions(rsr_dir:str) -> dict:
    """When pointed to a directory with relative spectral response functions, makes a dictionary with the read-in data and sensors as keys.
    
    Args: 
        directory (str): Relative path to the directory with RSR function definition files as .csv
    
    Returns:
        dict: Dictionary with RSR filenames as keys and RSR functions as Dataframe values.
    """
    
    functions = {}
    for file in glob.glob(os.path.join(rsr_dir, "*.csv")):
        #remove the .csv extension
        filename = os.path.basename(file)[:-4]
        # read in the data
        data = pd.read_csv(file, index_col=0)
        data = data.dropna(axis = 1, how = "all")
        functions[filename] = data
    return functions

def resample_spectra(spectra:pd.DataFrame, rsr_function:pd.DataFrame, minimum_response_trim = 0.0) -> pd.DataFrame:
    """
    Spectrally resample spectra to a relative spectral response (RSR).
    Args:
        spectra (pd.DataFrame): Full resolution spectra: samples as rows, wavelengths as cols.
        rsr_function (pd.DataFrame): RSR: wavelengths as rows, bands as cols.
        minimum_response_trim (float): 0-1. If set, trims bands that do not reach this minimum cumulative response. Value of 0.1 means that bands with less than 10% of their response covering the spectral range of the input data will be omitted from the resampled data.
    Returns:
        pd.DataFrame: Resampled data as a DataFrame where rows correspond to samples and columns to satellite sensor bands.
    """
   
    # Transpose to wavelengths as rows like rsr functions
    spectra = spectra.transpose()
    spectra.index = spectra.index.astype("int64")
    
    # Define wavelength range
    min_wavelength = min(spectra.index)
    max_wavelength = max(spectra.index)
   
    # Make cumsum array to find where to cut off resampled spectra
    cumu = np.cumsum(rsr_function.loc[min_wavelength:max_wavelength, :], axis = 0)
    min_response_mask = cumu.max(axis = 0) > minimum_response_trim
    
    # Calculate the total sensor response per band
    response_denom = np.sum(rsr_function, axis = 0)
   
    # Create placeholder for resampling results
    n_samples = len(spectra.columns)
    resampled_data = np.zeros(shape=(n_samples, len(response_denom)), dtype=float)
   
    # Resample each spectrum
    for n in range(n_samples):
        one_sample = spectra.iloc[:, n]
        signal = rsr_function.mul(one_sample, axis=0)
        signal_numerators = signal.sum()
        resampled_data[n] = signal_numerators / response_denom
    
    # Convert to DataFrame
    resampled_data = pd.DataFrame(resampled_data, columns=rsr_function.columns, index=spectra.columns)
    
    # Trim bands that do not reach the minimum response threshold
    resampled_data = resampled_data.loc[:, min_response_mask]
   
    print(f"Resampling data to {resampled_data.shape[1]} bands")
   
    return resampled_data

def export_resampled_data(resampled_data, sensor_name, out_dir, prefix = None):
    """
    Exports the resampled data to a CSV file.

    Args:
        Resampled_data (np.ndarray): The resampled data to be saved.
        sensor_name (str): The name of the sensor being resampled to.
        output_directory (str): Where to save the resampled data.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Define output file path
    if prefix: 
        output_file = os.path.join(out_dir, f"{prefix}_{sensor_name}_resampled.csv")
    else:
        output_file = os.path.join(out_dir, f"{sensor_name}_resampled.csv")

    # Save to CSV
    resampled_data.to_csv(output_file, index=True)
    
    print(f"Resampled data for {sensor_name} saved to {output_file}")

def resample_to_all(rsr_dir: str, spectra, out_dir, minimum_response_trim = 0, prefix = None):
    """
    Resamples spectra to all SRFs in a directory. Saves the resampled data to csv files.
    
    Args: 
        rsr_dir (str): Directory containing the relative spectral response functions (SRFs).
        spectra (pd.DataFrame): DataFrame containing the spectra to be resampled.
        out_dir (str): Directory where the resampled data will be saved.
    """
    rsr_functions = get_rsr_functions(rsr_dir)
    
    print(f"Dictionary contents: {list(rsr_functions.keys())}")
    print(f"About to iterate over {len(rsr_functions)} items")
    
    for sensor_name, rsr_function in rsr_functions.items():
        print(f"Resampling to {sensor_name}...")
        resampled_data = resample_spectra(spectra, rsr_function, minimum_response_trim = minimum_response_trim)
        export_resampled_data(resampled_data, sensor_name, out_dir, prefix= prefix)
        
    for file in glob.glob(os.path.join(out_dir, "*.csv")):
        #remove the .csv extension
        filename = os.path.basename(file)[:-4]
        print(f"Successfully saved resampled data for {filename}")
    print("Resampling completed.")
    