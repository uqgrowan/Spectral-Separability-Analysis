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
        functions[filename] = data
    return functions

def resample_spectra(spectra:pd.DataFrame, rsr_function:pd.DataFrame) -> np.ndarray:
    """
    Spectrally resample spectra to a relative spectral response (RSR).

    Args:
        spectra (pd.DataFrame): Full resolution spectra: samples as rows, wavelengths as cols.
        rsr_function (pd.DataFrame): RSR: wavelengths as rows, bands as cols.

    Returns:
        np.ndarray: Resampled data as a 2D numpy array where rows correspond to samples and columns to satellite sensor bands.
    """
    
    # Transpose to wavelengths as rows like rsr functions
    spectra = spectra.transpose()

    #match index formats of the SRFs
    spectra.index= spectra.index.astype("int64")
    n_samples = len(spectra.columns)

    #Define Wavelength range to use
    min_wavelength = min(spectra.index)
    max_wavelength = max(spectra.index)

    # Trim RSR to wavelength range of the spectra
    resampling_SRF = rsr_function.loc[min_wavelength:max_wavelength, :]

    # Calculate the total sensor response per band
    response_denom = list(np.sum(resampling_SRF, axis = 0)) 
    
    #Create placeholder for resampling results
    resampled_data = np.zeros(shape=(n_samples, resampling_SRF.shape[1]),dtype=float)
    
    #resample each spectrum
    for n in range(n_samples):
        #testing multiplying the SRF dataframe by a single spectrum
        #extract one sample for testing
        one_sample = spectra.iloc[:, n]

        #multiply SRf dataframe by sample
        signal = resampling_SRF.mul(one_sample, axis=0)

        #sum by SRF band
        signal_numerators = signal.sum()

        #n=4 #index of column used, for labeling below
        #add entry to dictionary
        resampled_data[n] = (signal_numerators/response_denom).T
    resampled_data = pd.DataFrame(resampled_data)
    resampled_data.dropna(axis =1, how = "all",inplace = True)
    
    return resampled_data

def export_resampled_data(resampled_data, sensor_name, out_dir):
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
    output_file = os.path.join(out_dir, f"{sensor_name}_resampled.csv")

    # Save to CSV
    resampled_data.to_csv(output_file, index=False)
    
    print(f"Resampled data for {sensor_name} saved to {output_file}")

def resample_to_all(rsr_dir: str, spectra, out_dir):
    """
    Resamples spectra to all SRFs in a directory. Saves the resampled data to csv files.
    
    Args: 
        rsr_dir (str): Directory containing the relative spectral response functions (SRFs).
        spectra (pd.DataFrame): DataFrame containing the spectra to be resampled.
        out_dir (str): Directory where the resampled data will be saved.
    """
    rsr_functions = get_rsr_functions(rsr_dir)
    
    for sensor_name, rsr_function in rsr_functions.items():
        print(f"Resampling to {sensor_name}...")
        resampled_data = resample_spectra(spectra, rsr_function)
        export_resampled_data(resampled_data, sensor_name, out_dir)
    print("Resampling completed for all sensors.")