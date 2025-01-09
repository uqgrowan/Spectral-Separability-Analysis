#Spectral classification pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import glob
import os

def classification_pipeline(independent, dependent, n_comps: float, preprocessing = "standard scaler", reducer = "pca", classifier= "random forest classifier", test_split = 0.3):
	"""Run a classification pipeline including standardization, feature extraction and classification

	Args:
		Independent: Dataframe of spectral data
		dependent: 1 column Dataframe with class labels
		n_comps : number of components to keep
 	Outputs:
  		Dictionary with pipeline results of accuracy, y_test size, cumulative sum of variance, and confusion matrix. 
    
    """

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
	results = {}
	results["variance cum sum"] = pipe["reducer"].explained_variance_ratio_.cumsum()
	results["accuracy"] = pipe.score(x_test, y_test)
	predictions = pipe.predict(x_test)
	results["matrix"] = confusion_matrix(y_test, predictions, labels= pipe.classes_)
	results["y_test"] = y_test 
	return results

def many_runs(runs, n_comps, filename, column, sensor_keys):
	accuracies = []
	cm = np.zeros(shape = (len(dep.unique()), len(dep.unique())))
	cumsums = np.zeros(n_comps)
	for i in range (0, runs):  
		run = classification_pipeline(indep, dep, n_comps)
		accuracies.append(run["accuracy"])
		cm += run["matrix"]
		cumsums += run["variance cum sum"]
		print(i)
	print(f"Mean accuracy for this pipeline is {sum(accuracies)/runs}.")
	print(f"The cumulative component contributions are {cumsums/runs}.")
	class_totals = pd.DataFrame(run["y_test"].value_counts()).sort_index()
	testing_class_sizes = np.array(class_totals["count"].to_list()) 
	conf_mat = (cm/runs/testing_class_sizes.reshape(-1,1))
	sorted_labels = dep.sort_values().unique()
	disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=sorted_labels)
	disp.plot(cmap = "Blues", xticks_rotation= 45, values_format = ".2f", ax = plt.gca())
	plt.tight_layout()  # Adjust layout to prevent label cutoff
	plt.savefig(f"./figure_{column}_{sensor_keys[filename]}.svg", bbox_inches='tight', dpi=300)
	plt.show()
	return accuracies

##################################################################
#parameterize the variables
labels = pd.read_csv(r'Combined_analysis\All_reflectance_labels.csv')
dep = labels.iloc[:,1]	#identify which class scheme to use by column number

indep = pd.read_csv(r"Combined_analysis\MEL_NZ_TAS_spectra.csv").drop(["Class", 'setup', 'site'], axis = 1)
n_comps = 3

##################################################################
# Get single run accuracy
accuracy = classification_pipeline(indep, dep, n_comps, reducer= 'pca')
print(accuracy)

##################################################################
# Run many times and average
accuracy_100 = many_runs(10,3, "dove.csv", "0", sensor_keys)

##################################################################
# Run many times across all 

#load table of n_comps and limit to rows found by proximity method
n_comps_table = pd.read_csv(r"Optimal_n_overview.csv", index_col=0)[:5]

#Name lookup dictionary
sensor_keys = {
    'dove.csv': 'dov',
    'asd_MEL_NZ_TAS_spectra.csv': 'asd',
    'landsat_9.csv': 'lan',
    'sentinel_2.csv': 'sen',
    'skysat.csv': 'sky',
    'superdove.csv': 'sup'
}

#make results holder dictionary
results= {}

#loop over grouping coloumns and all files in resampling folder
for column in range(0, labels.shape[1]):
    dep = labels.iloc[:,column]	#identify which class scheme to use by column number
    for filepath in glob.iglob(r'.\Combined_analysis\Resampled\*.csv'):
        filename = os.path.basename(filepath)
        n_comp = n_comps_table.loc[column, sensor_keys[filename]]
        indep = pd.read_csv(filepath).drop(["Class", 'setup', 'site'], axis = 1)
        results[f"{column}_{sensor_keys[filename]}"] = many_runs(100, n_comp, filename, column, sensor_keys)
        
results = [np.mean(results[key]) for key in results]
results_df = pd.DataFrame.from_dict(results).mean().to_frame().T
results_df.to_csv(r".\RFC_pipeline_results.csv")