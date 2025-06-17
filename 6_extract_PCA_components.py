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
label_path = r'Combined_analysis\All_reflectance_labels.csv'
labels = pd.read_csv(label_path)
dep = labels.iloc[:,1]	#identify which class scheme to use by column number
data_path = r"Combined_analysis\MEL_NZ_TAS_spectra.csv"
indep = pd.read_csv(data_path).drop(["Class", 'setup', 'site'], axis = 1) #make independent variables data without class labels

# Run feature extraction pipeline
components_matrix = feature_extraction_components(dep, indep)

# Reformat results and make a column for wavelengths and make wavelengths start at min not 0
components_df = pd.DataFrame(components_matrix).transpose().reset_index(names = "wavelength")
components_df["wavelength"] = components_df["wavelength"] + min([int(i) for i in indep.columns])

# Plot components all on one plot
for column in components_df.columns[1:]:
    sns.lineplot(data = components_df, x = "wavelength", y= column, label = f" PCA {column+1}", markers= False, alpha = 1)
plt.xlabel("Wavelength (nm)")
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