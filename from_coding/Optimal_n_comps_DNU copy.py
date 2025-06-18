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

def optimal_n_comp_by_averages (melted, threshold, interval, maximum, filepath, c):
    """ Fitting a line then setting the derivative to 0 relies on the curve being fit to be a parabolic curve, or somethign with an maximum accuracy. For the optimal integer n-comps, the accuracies produce a set of data with rapid increase and then sudden plateau, closest to a log function. The previously described method would therefore not work. 
    I am therefore proposing to iteratively calculate group averages of accuracy with an additional n_comp value dropped each time. So the first time, calculate the average of the whole group, then calculate the average accuracies of using 2 through 19 components. If the latter is larger, n_comps = 1 is not optimal, and run it again with 2 components. Obviously, this will require a threshold of similarity to account for stochasticity. """
    results = {}
    results["melted"] = melted

    testing_data = melted
    highest_component = min(np.max(testing_data["components"])+1, 13)
    optimal_found = False
    while not optimal_found: 
        if testing_data.empty and threshold < maximum:
            threshold += interval
            testing_data = melted
            #print("reset testing data and upped the threshold.")
        elif threshold >= maximum : 
            print(f"No optimal n_comps found before maximum thresh reached.")
            break
        else:
            for t in range(1, highest_component):
                test_melt = testing_data[testing_data["components"] == t]
                test_average = np.mean(test_melt["value"])
                remaining_group = testing_data[testing_data["components"] != t]
                group_average = np.mean(remaining_group["value"])
                if group_average - test_average < threshold:
                    print(f"the optimal n_comps for {filepath} and grouping {c} is {t} and produces an average accuracy of {test_average}\n Threshold: {threshold}")
                    results["file"] = filepath
                    results["grouping"] = c
                    results["optimal_n"] = t
                    results["accuracy_n"] = test_average
                    results["remainder_avg"] = group_average
                    results["threshold"] = threshold
                    optimal_found = True
                    testing_data = pd.DataFrame()
                    break
                else:
                    testing_data = testing_data[testing_data["components"] > t]
                    #print("Optimal n_comps not yet found, moving on.")
        
    return results


# def optimal_n_comps_by_interval(melted,start_thresh = 0.01, interval = 0.005):
#     """Find the optimal number of n_comps by evaluating the improvement in average accuracy from adding an additional component. The threshold is the desired maximum improvement that can be foregone in exchange for keeping one less component. Starting threshold and increase interval can both be changed according to the dataset"""
#     keep_trying = True
#     highest_component = min(np.max(melted["components"]+1), 12)
#     results = {}
#     global filepath, c
#     while keep_trying:
#         for t in range(1, highest_component):
#             test_melt = melted[melted["components"] == t]
#             next_melt = melted[melted["components"] == t+1]
#             test_average = np.mean(test_melt["value"])
#             next_average = np.mean(next_melt["value"])
#             if next_average - test_average < start_thresh:
#                 print(f"the optimal n_comps for {filepath} and grouping {c} is:  {t} and produces an average accuracy of\n T average: {test_average} \n  The next average: {next_average} \n Accuracy interval : {start_thresh}")
#                 keep_trying = False
#                 results["file"] = filepath
#                 results["grouping"] = c
#                 results["optimal_n"] = t
#                 results["accuracy_n"] = test_average
#                 results["interval"] = start_thresh
#                 break
#             else:
#                 melted = melted[melted["components"] > t]
#                 start_thresh += interval
#     return results

def optimal_n_comps_by_interval(melted, start_thresh=0.01, interval=0.005):
    """Find the optimal number of n_comps by evaluating the improvement in average accuracy from adding an additional component."""
    keep_trying = True
    highest_component = min(np.max(melted["components"]) + 1, 13)
    results = {}
    global filepath, c
    
    while keep_trying:
        found_optimal = False  # Flag to check if optimal value has been found in this iteration
        for t in range(1, highest_component-1):
            test_melt = melted[melted["components"] == t]
            next_melt = melted[melted["components"] == t + 1]
            test_average = np.mean(test_melt["value"])
            next_average = np.mean(next_melt["value"])
            
            if next_average - test_average < start_thresh:
                print(f"the optimal n_comps for {filepath} and grouping {c} is: {t} and produces \n T average: {test_average}\n The next average: {next_average}\n Accuracy interval: {start_thresh}")
                results["file"] = filepath
                results["grouping"] = c
                results["optimal_n"] = t
                results["accuracy_n"] = test_average
                results["interval"] = start_thresh
                found_optimal = True
                break
        if found_optimal:
            keep_trying = False
        else:
            start_thresh += interval
            
    return results

#import data
data_path = r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\MEL_NZ_TAS_spectra.csv' 
spectra = pd.read_csv(data_path)
label_path = r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\All_reflectance_labels.csv'
labels = pd.read_csv(label_path, index_col= 0)

#combine into one dataframe
labelled_spectra = pd.concat([labels, spectra.iloc[:, 1:]], axis = 1)

#Clean for only handheld spectra of targets
bad_classes = ["wr", "bryozoans", "background", "stray_light", "bad"]
bad_setup = ["sphere"]
labelled_data = labelled_spectra[~labelled_spectra["Class"].isin(bad_classes)]
labelled_data = labelled_data[~labelled_data["setup"].isin(bad_setup)]

#identify which class scheme to use by column number
labels = labelled_data.select_dtypes("object")
dep = labels.iloc[:, 5]

#make independent variables data without class labels
indep= labelled_data.select_dtypes("number") 
indep = indep.dropna(axis = 1, how = "any")

###Get accuracy over many runs for many different n_comps values to see what is optimal, output as dictionary then plotted
iterations = range(900, 1000, 2)  #Fractional n_comps
#iterations = range(1, 10)   #Integer n_comps
accuracy_assessment = {}
for s in iterations:
	n = s/1000   #only need for fractional n_comps DO NO USE FOR INTEGER N_COMPS
	#n = s   #only for integer n_Comps, not for fractional
	accuracies = []
	for i in range (0, 100):  #100 runs is about 15 secsond to run for PCA, 12 for kPCA
		run = feature_extraction_pipeline(indep, dep, n)
		accuracies.append(run)
	accuracy_assessment[n] = accuracies
		
assess_df = pd.DataFrame.from_dict(accuracy_assessment, orient = 'index')
components = accuracy_assessment.keys()
assess_df["components"] = components
melted = assess_df.melt(id_vars = "components")
assess_df.head()
print(melted["value"])
assess_df.head()
sns.scatterplot(data = melted, x = "components", y = "value", alpha = 0.15)
plt.show()

#melted.to_csv("optimizing_n-comps_pca_rfc_kbgrm_accuracies_100_runs.csv")

#determining optimal integer n_comps 
melted.head() 
optimal_n_comp_by_averages(melted, 0.05, 0.005, 0.1,"data/all_targets_normalized550_dove_SRF.csv",1)
optimal_n_comps_by_interval(melted, 0.01, 0.005)


#Iterate over all resampled files and label columns

label_path = r'C:\Users\s4770224\Documents\coding\Spectral analysis\all_targets_labels.csv'
labels = pd.read_csv(label_path)
labels_no_wr = labels[labels["Class"]!= "bryozoans"]

all_optimizations = {}
for filepath in glob.iglob('data/all_targets_normalized55*.csv'):
    spectra = pd.read_csv(filepath, index_col=[0])
    spectra_targets = spectra[spectra["Class"] != "bryozoans"]  #Remove bryos from target set
    for c in range(1, 6):
        dep = labels_no_wr.iloc[:,c]	#identify which class scheme to use by column number
        indep= spectra_targets.drop(["Class", 'setup', 'site'], axis = 1)
        indep = indep.iloc[:, 1:]
        
        ###Get accuracy over many runs for many different n_comps values to see what is optimal
        iterations = range(1, min(len(indep.columns), 20))   #Integer n_comps
        accuracy_assessment = {}
        for s in iterations:
            #n = s/1000   #only need for fractional n_comps DO NO USE FOR INTEGER N_COMPS
            accuracies = []
            for i in range (0, 100):  #100 runs is about 15 secsond to run for PCA, 12 for kPCA
                run = feature_extraction_pipeline(indep, dep, s)
                accuracies.append(run)
            accuracy_assessment[s] = accuracies
                
        assess_df = pd.DataFrame.from_dict(accuracy_assessment, orient = 'index')
        components = accuracy_assessment.keys()
        assess_df["components"] = components
        melted = assess_df.melt(id_vars = "components")           
        file_results = optimal_n_comp_by_averages(melted, 0.005, 0.005,0.05, filepath, c)
        run_name = filepath + "_" + str(c)
        all_optimizations[run_name] = file_results

print(all_optimizations.keys())


dep = labels_no_wr.iloc[:,5]	#identify which class scheme to use by column number
print(dep.unique())

opts_df = pd.DataFrame.from_dict(all_optimizations['data\\all_targets_normalized550_skysat_SRF.csv_2']["melted"])
opts_df.head()
opts_melts = opts_df.iloc[2,:]
opts_melts.head()