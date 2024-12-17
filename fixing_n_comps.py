### File contains:  1. code block for running and plotting accuracies for one sensor and grouping scheme.
###                 2. function for calculating many runs over all sensors and all grouping schemes.
###                 3. function for selecting optimal n_comps by margin gain
###                 4. code block for selecting optimal n_comps by proximity to maximum possible accuracy


#Determining optimal n_comps to extract
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
	#print(f"the testing accuracy is {pipe.score(x_test, y_test)}. The training accuracy is {pipe.score(x_train, y_train)}")
	##print(pipe["reducer"].components_)
	# results = {}
	# results["variance cum sum"] = pipe["reducer"].explained_variance_ratio_.cumsum()
	# results["accuracy"] = pipe.score(x_test, y_test)
	# predictions = pipe.predict(x_test)
	# results["matrix"] = confusion_matrix(y_test, predictions, labels= pipe.classes_)
	# results["y_test"] = y_test 
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

def test_threshold_by_interval(means_df, sensor, thresh, print_updates = False):
    #make results df holder
    results_test_interval = pd.DataFrame()
    
    for col in means_df.columns:
        for t in range(1, means_df.shape[0]+1):
            d=t+1
            if t == means_df.shape[0]:
                results_test_interval[col] = [sensor, float(t), float(means_df.loc[t, col]), float(10000)] #list(output_list)
                if print_updates:
                    print(f"All components needed to satisfy threshold value of {thresh}.")
                break
            else:
                if (means_df.loc[d, col] - means_df.loc[t, col]) < thresh:
                    #output_list= [float(t), float(mean_all_schemes.iloc[t, col]), float(mean_all_schemes.iloc[d, col])]
                    results_test_interval[col] = [sensor, float(t), float(means_df.loc[t, col]), float(means_df.loc[d, col])] #list(output_list)
                    if print_updates:
                        print(f"Optimal components for col {col} = {t}.")
                    break
    return results_test_interval

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

################################################################################################################################################################

#Define labels and number of grouping schemes
labels = pd.read_csv(label_path)
num_categories = len(labels.columns)
labels.shape


#Make results placeholders for means of all runs and all raw runs
means_dict = {}
all_accuracies_dict = {}

spectra = pd.read_csv(r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\MEL_NZ_TAS_spectra.csv') #, index_col=[0])
indep= spectra.drop(["Class", 'setup', 'site'], axis = 1)
indep.shape

#For loop to iterate over all resampled files
for filepath in glob.iglob(r'.\Combined_analysis\MEL_NZ_TAS_spectra.csv'):
    spectra = pd.read_csv(filepath) #, index_col=[0])
    indep= spectra.drop(["Class", 'setup', 'site'], axis = 1)
    filename = os.path.basename(filepath)
    
    #Accuracies holder df
    all_schemes_accuracy = {}
    
    #Iterate over all grouping schemes in label set
    for c in range(0,num_categories):
        dep = labels.iloc[:,c]	#identify which class scheme to use by column number
        
        #define range of possible n_comps values to check
        n_comps_range = range(1, min(len(indep.columns), 15))   #Integer n_comps
        
        #make dictionary for accuracy results
        accuracy_assessment = {}
        
        #test over n_comps_range for many iterations
        for s in n_comps_range:
            #n = s/1000   #only need for fractional n_comps DO NO USE FOR INTEGER N_COMPS
            accuracies = []
            for i in range (0, 100):
                run = feature_extraction_pipeline(indep, dep, s)
                accuracies.append(run)
            accuracy_assessment[s] = accuracies

        #Store results
        all_schemes_accuracy[c] = accuracy_assessment
    
    #convert results dictionary to dataframe
    all_schemes_df = pd.DataFrame.from_dict(all_schemes_accuracy)
    
    #Make DF of average accuracies for all categories and N_comps values
    mean_all_schemes = all_schemes_df.map(np.mean)
    means_dict[filename] = mean_all_schemes
    all_accuracies_dict[filename] = all_schemes_df
    print(f" Done calculating means df for {filename}." )

################################################################################################################################################################
#Choose n optimal by marginal gain
interval_results = pd.DataFrame()
for key in means_dict.keys():
    interval_results = pd.concat([interval_results, test_threshold_by_interval(means_dict[key], key, 0.05)], axis = 1)
interval_results
interval_results.to_csv("Optimal_n_by_marginal_gain.csv")
################################################################################################################################################################
#Choose n optimal by proximity to maximum accuracy
maximum_results = pd.DataFrame()
maximum_proximity_results = {}
threshold = 0.10
for key in means_dict.keys():
    min_index = []
    for col in means_dict[key].columns:
        df = means_dict[key]
        mask = df[col] > (np.max(df[col]) - threshold)
        filtered_df = df[col][mask]
        smallest_index = filtered_df.index.min()
        min_index.append(smallest_index)
    maximum_proximity_results[key] = min_index
    
max_prox_results=pd.DataFrame.from_dict(maximum_proximity_results)
max_prox_results.to_csv("Optimal_n_by_max_proximity.csv")
max_prox_results