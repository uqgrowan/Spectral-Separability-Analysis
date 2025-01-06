import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import glob
import os

def calculate_kaiser_value(explained_variance):
    """Calculate Kaiser value: Number of components with eigenvalues > 1"""
    return np.sum(explained_variance > 1)

def find_elbow_point(explained_variance_ratio):
    """Find the elbow point using the kneedle algorithm"""
    kneedle = KneeLocator(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, curve='convex', direction='decreasing')
    return kneedle.elbow

def process_dataset(X, dataset_name):
    """Process a single dataset to calculate Kaiser value and elbow point"""
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_
    
    # Calculate Kaiser value
    kaiser_value = calculate_kaiser_value(explained_variance)
    
    # Find elbow point
    elbow_point = find_elbow_point(explained_variance_ratio)
    print(explained_variance_ratio[kaiser_value].cumsum())
    print(explained_variance[kaiser_value])
    
    return {
        'Dataset': dataset_name,
        'Kaiser Value': kaiser_value,
        'Elbow Point': elbow_point,
        'Explained variance - Kaiser' : explained_variance_ratio[0:kaiser_value].cumsum()[-1],
        'Explained variance - Elbow' : explained_variance_ratio[0:elbow_point].cumsum()[-1],
            }

def main():
    # Define datasets
    datasets = {
        'sen2': sen2,
        'lan9': lan9,
        'asd': asd,
        'sup': sks,
        'sks': sup,
        'dov': dov,
    }
    
    # Process each dataset
    results = []
    for name, data in datasets.items():
        result = process_dataset(data, name)
        results.append(result)
    
    # Create DataFrame with results
    df_results = pd.DataFrame(results)
    print(df_results)

    #Optionally, save results to CSV
    df_results.to_csv('pca_results_optimal_n_comps.csv', index=False)

sen2 = pd.read_csv(r'Combined_analysis\Resampled\sentinel_2.csv')
sen2 = sen2.drop(["Class", "setup", "site"], axis = 1)
sen2 = sen2.dropna(axis = 1, how = "any")
asd = pd.read_csv(r'Combined_analysis\Resampled\asd_MEL_NZ_TAS_spectra.csv')
asd = asd.drop(["Class", "setup", "site"], axis = 1)
asd = asd.dropna(axis = 1, how = "any")
lan9 = pd.read_csv(r'Combined_analysis\Resampled\landsat_9.csv')
lan9 = lan9.drop(["Class", "setup", "site"], axis = 1)
lan9 = lan9.dropna(axis = 1, how = "any")
sks = pd.read_csv(r'Combined_analysis\Resampled\skysat.csv')
sks = sks.drop(["Class", "setup", "site"], axis = 1)
sks = sks.dropna(axis = 1, how = "any")
sup = pd.read_csv(r'Combined_analysis\Resampled\superdove.csv')
sup = sup.drop(["Class", "setup", "site"], axis = 1)
sup = sup.dropna(axis = 1, how = "any")
dov = pd.read_csv(r'Combined_analysis\Resampled\dove.csv')
dov = dov.drop(["Class", "setup", "site"], axis = 1)
dov = dov.dropna(axis = 1, how = "any")

for filepath in glob.iglob(r'.\Combined_analysis\Resampled\*.csv'):
    filename = os.path.basename(filepath)[0:3]
    spectra = pd.read_csv(filepath)
    indep= spectra.drop(["Class", 'setup', 'site'], axis = 1)
    filename = os.path.basename(filepath)
main()