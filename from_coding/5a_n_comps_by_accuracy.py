### File contains:  1. code block for running and plotting accuracies for one sensor and grouping scheme.
###                 2. function for calculating many runs over all sensors and all grouping schemes.
###                 3. function for selecting optimal n_comps by margin gain
###                 4. code block for selecting optimal n_comps by proximity to maximum possible accuracy

import pandas as pd
import numpy as np
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

def classification_pipeline(independent, dependent, n_comps: float, test_split = 0.3, preprocessing = "standard scaler", reducer = "pca", classifier= "random forest classifier"):
	"""
    Run a classification pipeline (standardization, feature extraction and classification). Outputs the accuracy of the pipeline on a testing subset. Functions available to the pipeline are defined below in the dictionary 'functions' to allow them to be callable. *** Note that other functions need to be imported before use.*** 
 
    Args:
        independent: DataFrame with spectral data
        dependent: Labels Dataframe
        n_comps: number of PCA components to include in classification input data
        test_split: (float) fraction of total dataset to be kept for testing
        preprocessing: Standardization fucntion to use
        reducer: Feature extraction function to use
        classifier: Classification function to use
    
    Returns: 
        Pipeline accuracy (float)
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
	return pipe.score(x_test, y_test)

def find_min_indices_near_max(df, threshold=0.05):
    """
    Find the smallest index for each column where the value is within threshold of the maximum.
    
    Args:
        df: pandas DataFrame
        threshold: float, acceptable distance from maximum (default: 0.05)
    
    Returns:
        list of minimum indices
    """
    return [(df[col][df[col] > (df[col].max() - threshold)]).index.min() 
            for col in df.columns]

def test_threshold_by_interval(means_df, sensor, thresh, print_updates=False):
    """
    Find minimum number of components n where the accuracy marginally gained by including n+1 components is less than the threshold value. 
    
    Args:
        means_df: DataFrame with component and value columns
        sensor: string identifier for the sensor
        thresh: threshold value for comparison
        print_updates: boolean to control printing of updates
    
    Returns:
        DataFrame with results for each column
    """
    # Create copy of relevant columns
    df = means_df.iloc[:, 1:].set_index("component")
    
    # Initialize results dictionary
    results = {}
    
    for col in df.columns:
        # Calculate differences between consecutive rows
        diff = df[col].diff()
        
        # Find first index where difference is less than threshold
        # Shift by 1 to get the index where the condition starts
        mask = (diff.abs() < thresh)
        if mask.any():
            t = mask.idxmax()  # First True index
            results[col] = [
                sensor,
                float(t-1),  # Subtract 1 to get the starting component
                float(df.loc[t-1, col]),  # Value at t
                float(df.loc[t, col])     # Value at t+1
            ]
        else:
            # Case where threshold is never met
            max_idx = df.shape[0]
            results[col] = [
                sensor,
                float(max_idx),
                float(df.loc[max_idx, col]),
                float(10000)  # Indicator that all components were needed
            ]
            
        if print_updates:
            if mask.any():
                print(f"Optimal components for col {col} = {t-1}")
            else:
                print(f"All components needed to satisfy threshold value of {thresh}.")
    
    # Convert results to DataFrame all at once
    return pd.DataFrame.from_dict(results)

def round_dict_values(d, decimals=4):
    """
    Recursively rounds all numerical values in a dictionary to specified decimal places.
    Works with nested dictionaries and lists.
    """
    if isinstance(d, dict):
        return {key: round_dict_values(value, decimals) for key, value in d.items()}
    elif isinstance(d, list):
        return [round_dict_values(value, decimals) for value in d]
    elif isinstance(d, (int, float)):
        return round(d, decimals)
    else:
        return d

################################################################################################################################################################
# Define labels and number of grouping schemes
labels = pd.read_csv(r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\New_reflectance_labels.csv')
num_categories = len(labels.columns)

# Make results placeholders for means of all runs and all raw runs
all_accuracies_dict = {}
all_means_dict= {}

sensor_keys = {
    'dove.csv': 'dov',
    'asd_MEL_NZ_TAS_spectra.csv': 'asd',
    'landsat_9.csv': 'lan',
    'sentinel_2.csv': 'sen',
    'skysat.csv': 'sky',
    'superdove.csv': 'sup'
}

# For loop to iterate over all resampled files
for filepath in glob.iglob(r'.\Combined_analysis\Resampled\*.csv'):
    indep= pd.read_csv(filepath).drop(["Class", 'setup', 'site'], axis = 1)
    filename = os.path.basename(filepath)
    #define range of possible n_comps values to check
    n_comps_range = range(1, min(len(indep.columns), 15))
    
    #Accuracies holder df
    all_schemes_accuracy = {}
    
    #Iterate over all grouping schemes in label set
    for c in range(4,6): #num_categories):
        dep = labels.iloc[:,c]	#identify which class scheme to use by column number
        
        #Calculate dictionary of accuracy results    
        accuracy_assessment = {
            s: [classification_pipeline(indep, dep, s) for _ in range(100)]
            for s in n_comps_range
        }
        #Store results
        all_schemes_accuracy[c] = accuracy_assessment
    
    #convert results dictionary to dataframe
    all_schemes_df = pd.DataFrame.from_dict(all_schemes_accuracy)
    
    #Make DF of average accuracies for all categories and N_comps values
    mean_all_schemes = all_schemes_df.map(np.mean)
    all_accuracies_dict[sensor_keys[filename]] = all_schemes_df
    all_means_dict[sensor_keys[filename]] = mean_all_schemes
    print(f" Done calculating means df for {filename}." )

combo_means = pd.concat(all_means_dict, names=["sensor", "component"]).reset_index()
combo_means.to_csv("combo_order_means.csv")

rounded_dict = round_dict_values(all_accuracies_dict, 4)

################################################################################################################################################################
# Choose n optimal by marginal gain
interval_results = pd.DataFrame()
for satellite in combo_means["sensor"].unique():
    means = combo_means[combo_means["sensor"] == satellite]
    interval_results = pd.concat([interval_results, test_threshold_by_interval(means, satellite, 0.05)], axis = 1)
interval_results.to_csv("Optimal_n_by_marginal_gain_order.csv")
interval_results

################################################################################################################################################################
# Choose n optimal by proximity to maximum accuracy for all means dictionary entries
maximum_proximity_results = {
    key: find_min_indices_near_max(df, threshold=0.05)
    for key, df in all_means_dict.items()}
maximum_results = pd.DataFrame.from_dict(maximum_proximity_results, orient='index')
maximum_results.to_csv("Optimal_n_by_max_proximity_order.csv")
maximum_results