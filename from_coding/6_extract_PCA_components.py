#Spectral classification pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline

def feature_extraction_components(dep, indep, preprocessing = "standard scaler", reducer = "pca", test_split = 0.3, n_comps = 5):
	"""Extract the PCA components for a spectral dataset. Pipeline includes standardization and feature extraction (option for PCA or kPCA).  
	
    Args: 
		dep: class labels in a Dataframe
		independent: Dataframe of spectra
		test_split: fraction of data to keep for testing in any given run, default is 0.3
		n_comps: number of PCA components to compute, default is 5
		preprocessing: str identifying which standardizer to use. Only StandardScaler is included but others can be added.
		reducer: str identifying what fetaure extraction method to use, default PCA but kPCA also available
  
	Output:
		numpy array of n components with band contribution values
	"""

	x_train, x_test, y_train, y_test = train_test_split(indep, dep, test_size= test_split) 

	functions = {
		"pca" : PCA(n_components = n_comps),
		"kpca" : KernelPCA(n_components = n_comps),
		"standard scaler" : StandardScaler(),
	}
	components_pipe = Pipeline ( [ 
		("scaler", functions[preprocessing]),  #scale your data to mean 0, var 1
		("reducer", functions[reducer] ),    # extract first n PCA components. 0<n<1 will keep components until that variance is met (i.e. 0.9 = 90%)
		] )
	components_pipe.fit(x_train, y_train)
	return components_pipe["reducer"].components_


#parameterize the variables
label_path = r'Combined_analysis\New_reflectance_labels.csv'
labels = pd.read_csv(label_path)
dep = labels.iloc[:,1]	#identify which class scheme to use by column number
data_path = r"Combined_analysis\Resampled\asd_MEL_NZ_TAS_spectra.csv"
indep = pd.read_csv(data_path).drop(["Class", 'setup', 'site'], axis = 1) #make independent variables data without class labels
full_wavelengths = pd.DataFrame(indep.columns.astype("int"), columns= ["wavelength"])

data = pd.read_csv(r'Combined_analysis\MEL_NZ_TAS_spectra.csv')
labels = data.loc[:,"Class"]
labels["code"] = [labels.loc[i, "Class"] + " _" + labels.loc[i, "site"] for i in labels.index]
asd = asd.drop(["Class", "setup", "site"], axis = 1)
mask = asd.columns.isin([str(i) for i in range(755, 770)])
asd.loc[:, mask] = np.nan
indep = asd.dropna(axis = 1, how = "any")
filename = "asd_macros"



#Mask out bad bands 
bad_bands_list = [str(num) for num in range(753, 769)]
mask = indep.columns.isin(bad_bands_list)
indep.loc[:, mask] = np.nan
indep = indep.dropna(axis =1, how = "any")
wavelengths_masked = indep.columns.astype("int32")
wavelengths_masked

# Run feature extraction pipeline
components_matrix = feature_extraction_components(dep, indep, n_comps = 7)
components_df = pd.DataFrame(components_matrix).transpose()
components_df["wavelength"] = wavelengths_masked
components_all_wavelengths = pd.merge(components_df, full_wavelengths, how = "outer", on = "wavelength")
components_all_wavelengths

components_all_wavelengths.dropna()
# Plot components all on one plot
for column in components_df.columns[:-1]:
    sns.lineplot(data = components_all_wavelengths, x = "wavelength", y= column, label = f" PCA {column+1}", markers= False, alpha = 1)
    #sns.lineplot(data = components_df, x = components_df.index, y= column, label = f" PCA {column+1}", markers= False, alpha = 1)
plt.xlabel("Wavelength (nm)")
#plt.xlabel("Band Number")
plt.ylabel("Contribution to Component")
plt.ylim(-0.1,0.1)
plt.legend()
plt.show()

# Plot them on a facet grid
comps_melted = components_df.melt(id_vars = "wavelength", var_name = "component", value_name = "contribution")
comps_melted_8 = comps_melted[comps_melted["component"] <8]
sns.relplot(data=comps_melted_8, x="wavelength", y="contribution", col="component", col_wrap = 4, kind="line")
plt.tight_layout()
plt.show()