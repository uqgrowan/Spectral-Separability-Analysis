#Spectral classification pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score,balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
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
	all_classes = dependent.unique()
	x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size= test_split, stratify = dependent) 

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
	predictions = pipe.predict(x_test)
 
	results = {}
	results["variance cum sum"] = pipe["reducer"].explained_variance_ratio_.cumsum()
	results["accuracy"] = pipe.score(x_test, y_test)
	results["matrix"] = confusion_matrix(y_test, predictions, labels= all_classes, normalize = 'true')
	results["y_test"] = y_test
	results["recall"] = recall_score(y_test, predictions, average = "weighted")
	results["precision"] = precision_score(y_test, predictions, average = "weighted", zero_division=np.nan)
	results["bal_acc"] = balanced_accuracy_score(y_test, predictions) 
	return results

def many_runs(runs, n_comps, filename, column, sensor_keys, indep, dep):
	accuracies = []
	recalls = []
	precisions = []
	bal_accs = []
	cm = np.zeros(shape = (len(dep.unique()), len(dep.unique())))
	cumsums = np.zeros(n_comps)
	for i in range(0, runs):  
		run = classification_pipeline(indep, dep, n_comps)
		accuracies.append(run["accuracy"])
		recalls.append(run["recall"])       
		precisions.append(run["precision"]) 
		bal_accs.append(run["bal_acc"])     		
		cm += run["matrix"]
		cumsums += run["variance cum sum"]
        
	print(f"Mean accuracy for {sensor_keys[filename]} and grouping {column} is {np.mean(accuracies)}.")
	print(f"Mean recall for {sensor_keys[filename]} and grouping {column} is {np.mean(recalls)}.")
	print(f"Mean precision for {sensor_keys[filename]} and grouping {column} is {np.mean(precisions)}.")
	print(f"Mean balanced accuracy for {sensor_keys[filename]} and grouping {column} is {np.mean(bal_accs)}.")
	cm = (cm/runs)
	cm_df = pd.DataFrame(cm)
	cm_masked = cm_df.map(lambda v: str(round(v, 2)) if v>0 else "")
	plt.close()
	sns.heatmap(cm, annot=cm_masked, fmt = "s", cmap='Blues', 
                xticklabels= dep.unique(), 
                yticklabels= dep.unique(),
                annot_kws={"size" : 6}, 
                vmin = 0, 
                vmax = 1
                )
	plt.tight_layout(pad = 4, w_pad= 1, h_pad= 1)
	plt.xlabel('Predicted Labels')
	plt.xticks(rotation = 90)
	plt.ylabel('True Labels')
	plt.title(f'Confusion Matrix for {sensor_keys[filename]} and grouping {column}')
	plt.savefig(rf"C:\Users\s4770224\Documents\coding\Spectral_analysis\Plots\For_publication\RFC_cms\RFC_{sensor_keys[filename]}_{column}.svg")
	plt.show()
	return accuracies

def assign_targets_list(group = "all"):
    if group == 'kelp':
        target_classes = ['ecklonia', 'macrocystis', 'undaria', 'petalonia']
    elif group == "all":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'sand', 'rock', 'acrocarpia', 'hormosira', 'mussels','macrocystis', 'carpophyllum', 'gravel', 'grass', 'filamentous_rhodophyte', 'undaria', 'frondose_rhodophyte', 'shell_litter', 'ulva', 'sargassum', 'barnacle_shells', 'worm_castings', 'scytosiphon','petalonia']
    elif group == "browns":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'acrocarpia', 'hormosira', 'macrocystis', 'carpophyllum', 'undaria', 'sargassum', 'scytosiphon','petalonia']
    elif group == "farm": 
        target_classes = ['macrocystis', 'carpophyllum','filamentous_rhodophyte', 'undaria']
    return target_classes

def sort_samples(data):
    # Make sort_order of classes
	sort_order = ['ecklonia', 'macrocystis','undaria','phyllospora', 'durvillaea','hormosira','cystophora', 'acrocarpia', 'carpophyllum', 'sargassum', 'scytosiphon', 'petalonia', 'filamentous_rhodophyte', 'frondose_rhodophyte', 'ulva', 'grass','mussels','worm_castings', 'barnacle_shells', 'shell_litter','sand', 'gravel', 'rock']
 
	# Make a sort column with categorical type, sort to match the sort_order
	#data["sort_column"] = data["Class"]
	data['Class'] = pd.Categorical(data['Class'], categories=sort_order, ordered=True)
	data = data.sort_values('Class')
	return data
 
##################################################################
#parameterize the variables
labels = pd.read_csv(r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\New_reflectance_labels.csv')
many_run_results= {}
keep_classes = assign_targets_list("all") #could be all, kelp, browns, or farm 
#sorted_labels = sort_samples(labels)
class_mask = labels["Class"].isin(keep_classes) #make mask for selecting classes of interest 
column = 4
dep = sort_samples(labels).iloc[:,column]
#dep = sorted_labels.iloc[:,1][class_mask]	#identify which class scheme to use by column number and mask
filepath = "asd_MEL_NZ_TAS_spectra.csv"
filename = os.path.basename(r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\Resampled\asd_MEL_NZ_TAS_spectra.csv')
n_comp = 6
indep = sort_samples(pd.read_csv(r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\Resampled\asd_MEL_NZ_TAS_spectra.csv'))
#indep = sort_samples(indep)
indep = indep.drop(["Class", 'setup', 'site'], axis = 1) #.loc[class_mask]
many_run_results[f"{column}_{n_comps}"] = many_runs(100, n_comp, filename, column, sensor_keys, indep= indep, dep = dep)      


##################################################################
# Run many times and average
accuracy_100 = many_runs(10,3, "asd_MEL_NZ_TAS_spectra.csv", "0", sensor_keys= sensor_keys)

##################################################################
# Run many times across all 

#load table of n_comps and limit to rows found by proximity method
n_comps_table = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Outputs\n_comps_discussion\Optimal_n_overview.csv", index_col=0)[:6]

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
many_run_results= {}

#loop over grouping coloumns and all files in resampling folder
for column in range(4, 6): # labels.shape[1]):
    dep = sort_samples(labels).iloc[:,column] #.loc[class_mask]	#identify which scheme to use by column
    for filepath in glob.iglob(r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\Resampled\*.csv'):
        filename = os.path.basename(filepath)
        n_comp = int(n_comps_table.loc[column, sensor_keys[filename]])
        indep = sort_samples(pd.read_csv(filepath))
        #indep = sort_samples(indep)
        indep = indep.drop(["Class", 'setup', 'site'], axis = 1) #.loc[class_mask]
        many_run_results[f"{column}_{sensor_keys[filename]}"] = many_runs(100, n_comp, filename, column, sensor_keys, indep= indep, dep = dep)

        
results = [np.mean(results[key]) for key in many_run_results]
results_df = pd.DataFrame.from_dict(results).mean().to_frame().T
results
results_df.to_csv(r".\RFC_pipeline_results.csv")

results