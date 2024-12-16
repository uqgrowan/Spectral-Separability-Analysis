#Resampling to other SRFs

import pandas as pd
def choose_SRF(satellite, min_wavelength, max_wavelength):
    ''' set the Spectral response Function to use for resampling. Options are: sentinel_2, landsat_9, superdove, dove, skysat.'''
    if satellite == "sentinel_2":
        sentinel_2_SRF = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Spectral_Response_Functions\Sentinel2ABC_averaged_SPF.csv", index_col = "SR_WL")
        sentinel_2_SRF = sentinel_2_SRF.dropna(axis = 1, how = "any") #drop all wavelengths that are not measured in all sets
        sentinel_2_SRF = sentinel_2_SRF.loc[min_wavelength:max_wavelength,:]
        resampling_SRF = sentinel_2_SRF
    elif satellite == "landsat_9":
        landsat_9_SRF = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Spectral_Response_Functions\L9_OLI2_Ball_BA_RSR.v2-1.csv", index_col = "Wavelength")
        landsat_9_SRF.index= landsat_9_SRF.index.astype("int64")
        landsat_9_SRF = landsat_9_SRF.loc[min_wavelength:max_wavelength,:]
        resampling_SRF = landsat_9_SRF
    elif satellite == "superdove":
        superdove_SRF = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Spectral_Response_Functions\Superdove.csv" , index_col = "Wavelength (nm)")
        superdove_SRF.index= superdove_SRF.index.astype("int64")
        superdove_SRF = superdove_SRF.loc[min_wavelength:max_wavelength,:]
        resampling_SRF = superdove_SRF
    elif satellite == "dove":
        dove_SRF = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Spectral_Response_Functions\dove_r.csv", index_col = "Wavelength (nm)")
        dove_SRF.index= dove_SRF.index.astype("int64")
        dove_SRF = dove_SRF.loc[min_wavelength:max_wavelength,:]
        resampling_SRF = dove_SRF
    elif satellite == "skysat":
        skysat_SRF = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Spectral_Response_Functions\Skysat_RSR_Skysat14-SkySat19.csv", index_col = "Wavelength (nm)")
        skysat_SRF.index= skysat_SRF.index.astype("int64")
        skysat_SRF = skysat_SRF.loc[min_wavelength:max_wavelength,:]
        resampling_SRF = skysat_SRF
    else:
        print("Not a supported satellite. Try again.")
    return resampling_SRF

#Choose naming convention for output resampled data
in_file = "All_spectra"

#Import data
spectra = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\MEL_NZ_TAS_spectra.csv")

#drop any columns with nan values (i.e., limit to wavelengths covered by both sensors)
spectra = spectra.dropna(axis = 1, how = "any") #drop any nan columns

#make labels columns to re-append at end
front_cols = spectra.iloc[:,0:3]

#select only numerical data
spectra = spectra.select_dtypes(include = "number")
spectra = spectra.transpose()

#match index formats of the SRFs
spectra.index= spectra.index.astype("int64")
n_samples = len(spectra.columns)

#Define Wavelength range to use
min_wavelength = min(spectra.index)
max_wavelength = max(spectra.index)

#define the satellites of interest
satellites = ["sentinel_2", "landsat_9", "superdove", "dove", "skysat"]

#for loop to resample for each satellite
for satellite in satellites:
	#Create placeholder for results
	response_denom = []

	#calculate the total sensor response per band to be the denominator
	resampling_SRF = choose_SRF(satellite, min_wavelength, max_wavelength)
	for band in range(0, len(resampling_SRF.columns)):
		denom = resampling_SRF.iloc[:, band].sum()
		response_denom.append(denom)

	#Create placeholder for resampling results
	resampled_data = {}

	#resample each spectrum
	for n in range(0, n_samples):
		#testing multiplying the SRF dataframe by a single spectrum
		one_sample = spectra.iloc[:, n]  #extract one sample for testing
		signal = resampling_SRF.mul(one_sample, axis=0) #multiply SRf dataframe by that column
		signal_numerators = signal.sum() #sum by SRF band
		#n=4 #index of column used, for labeling below
		resampled_data[f"spectrum_{n}"] = (signal_numerators/response_denom) #add entry to dictionary
    #Make dataframe from dict and transpose
	resampled = pd.DataFrame(resampled_data).transpose()
    
    #Re-format dataframe for export
	resampled = resampled.reset_index(names="sample")
	resampled = resampled.dropna(axis = 1, how = "any")
	to_export = pd.concat([front_cols, resampled.drop(["sample"], axis = 1)], axis = 1)
 
	#output to csv file
	out_path = r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\Resampled\\"+ in_file + "_" + satellite +".csv"
	to_export.to_csv(out_path)