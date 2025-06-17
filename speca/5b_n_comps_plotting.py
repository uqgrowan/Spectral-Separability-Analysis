## Plot values accuracies resulting from differing values of n_comps


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import glob
import os

def feature_extraction_pipeline(independent, dependent, n_comps: float, test_split = 0.3, preprocessing = "standard scaler", reducer = "pca", classifier= "random forest classifier"):
	"""Run a feature extraction pipeline on a specified spectral dataset. Pipeline includes standardization, feature extraction and classification. Outputs the accuracy of the pipeline on a test dataset. Functions available to the pipeline are defined below int he dictionary 'functions' to allow them to be callable. *** Note that other functions need to be imported before use.*** """

	x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size= test_split, stratify = dep) 

	functions = {
		"logistic regression" : LogisticRegression(),
		"random forest classifier" : RandomForestClassifier(),
		"pca" : PCA(n_components = n_comps),
		"kpca" : KernelPCA(n_components = n_comps),
		"standard scaler" : StandardScaler(),
		"svc" : SVC()
	}
	pipe = Pipeline ( [ 
		("scaler", functions[preprocessing]),  #scale your data to mean 0, var 1
		("reducer", functions[reducer] ),    # extract n components. 0<n<1 will keep components until that variance is met (0.9 = 90%)
		( "classifier", functions[classifier] ), # train a random forest classifier
		] )
	pipe.fit(x_train, y_train)
	return pipe.score(x_test, y_test)

def import_format_data (data_path, label_path, col_num):
    """import and format data as neede for this script. Outputs two dataframes, one of independent variables (spectra) and one of dependent variables(labels). """
    spectra = pd.read_csv(data_path)
    labels = pd.read_csv(label_path)
    dep = labels.iloc[:,col_num]	#identify which class scheme to use by column number
    print(f"the unique values for scheme in column {col_num} are: {dep.unique()}")
    indep= spectra.drop(["Class", 'setup', 'site'], axis = 1) #isolate reflectance data 
    #indep = indep.iloc[:, 1:]
    indep = indep.dropna(axis = 1, how = "any")
    return dep, indep

data_path = r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\MEL_NZ_TAS_spectra.csv' 
label_path = r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\All_reflectance_labels.csv'
scheme_col_num = 4

dep, indep = import_format_data(data_path, label_path, scheme_col_num)

################################################################################################################################################################
###Get accuracy over many runs for many different n_comps values to see what is optimal, output as dictionary then plot it
iterations = range(1, 20)   #Integer n_comps
accuracy_assessment = {}
for s in iterations:
	accuracies = []
	for i in range (0, 100):
		run = feature_extraction_pipeline(indep, dep, s)
		accuracies.append(run)
	accuracy_assessment[s] = accuracies

#re-format iteration results for use		
assess_df = pd.DataFrame.from_dict(accuracy_assessment, orient = 'index')
components = accuracy_assessment.keys()
assess_df["components"] = components

#Make melted df where components are the number of PCA considered, variable is the run number and value is the accuracy
melted = assess_df.melt(id_vars = "components")

#Plot the iterations results
sns.scatterplot(data = melted, x = "components", y = "value", alpha = 0.15)
plt.show()