import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def run_pca(spectra, n_comps = 15):
    """
    Run PCA on the spectra data.
    Args:
        spectra (pd.DataFrame) : Reflectance data to be decomposed.
        
    Returns:
        tuple: Reflectance as components, explained variance ratio, and explained variance.
        
    Raises:
        ValueError: If the spectra contains non-numeric columns.
    """
    spectra = spectra.dropna(axis = 1, how = "any")
    if spectra.select_dtypes(include = "object").shape[1] >0:
        raise ValueError("Spectra data contains non-numeric columns. PCA requires numeric data only.")
    deco = PCA(n_comps)
    deco.fit(spectra)
    decomposed = deco.transform(spectra)
    comps = deco.components_
    return decomposed, deco.explained_variance_ratio_, deco.explained_variance_, comps

def find_elbow(explained_variance_ratio):
    """Find the elbow point using the kneedle algorithm"""
    kneedle = KneeLocator(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, curve='convex', direction='decreasing')
    return kneedle.elbow

def find_kaiser(explained_variance):
    """Calculate Kaiser value: Number of components with eigenvalues > 1"""
    return np.sum(explained_variance > 1)


class NCompsByAccuracy:
    
    def __init__(self, spectra, labels, labels_col="Class"):
        """
        Initialize the PCA decomposition class.
        """
        self.spectra = spectra
        self.labels = labels
        self.labels_col = labels_col
        self.accuracy_results = None


    def rf_pipeline(self,n_comps, test_split = 0.3):
        x_train, x_test, y_train, y_test = train_test_split(self.spectra, self.labels[self.labels_col], test_size= test_split, stratify = self.labels[self.labels_col])

        pipe = Pipeline ( [
            ("reducer", PCA(n_components=n_comps)),    # extract n components. 0<n<1 will keep components until that variance is met (0.9 = 90%)
            ( "classifier", RandomForestClassifier() ), # train a random forest classifier
            ] )
        pipe.fit(x_train, y_train)
        return pipe.score(x_test, y_test)


    def run_rfs(self, n, run_count = 100):
        """ Run the random forest pipeline a given number of times with redivision of the data every run, aggregating the results."""
        scores = []
        for _ in range(run_count):
            score = self.rf_pipeline(n_comps = n)
            scores.append(score)
        return scores


    def calc_mean_accuracies(self, max_n):
        """
        Calculate the mean accuracies for PCA components from 1 to max_n.
        
        Args:
            max_n (int): Maximum number of PCA components to consider.
            
        Returns:
            pd.DataFrame: DataFrame with mean accuracies for each number of components.
        """
        results = {}
        for n in range(1, max_n + 1):
            scores = self.run_rfs(n = n)
            results[n] = {'mean_accuracy': np.mean(scores), 'all accuracies' : scores}
        self.accuracy_results = results


    def find_n_by_max_proximity(self, threshold=0.05):
        """
        Find the smallest index for each column where the value is within threshold of the maximum.
        
        Args:
            df: pandas DataFrame
            threshold: float, acceptable distance from maximum (default: 0.05)
        
        Returns:
            list of minimum indices
        """
        df = pd.DataFrame.from_dict(self.accuracy_results)
        df = df.loc["mean_accuracy"]
        
        results = {}
        n = (df[df > (df.max() - threshold)]).index.min()
        results["n"] = n
        results["Accuracy at n"] = float(df.loc[n])
        results["Max Accuracy"] = float(df.max())
        return results


    def find_n_by_margin(self, thresh = 0.05, verbose =False):
        """
        Find minimum number of components n where the accuracy marginally gained by including n+1 components is less than the threshold value. 
        
        Args:
            thresh: threshold value for comparison
            verbose (Bool)
        
        Returns:
            DataFrame with results for each column
        """
        df = pd.DataFrame.from_dict(self.accuracy_results)
        df = df.loc["mean_accuracy"]
        # Calc marginal gains
        diff = df.diff()
        
        # Initialize results dictionary
        results = {}
        
        #Mask where marginal gain threshold is met
        mask = diff[diff.abs()<thresh]

        if mask.any():
            t = mask.idxmax()  # First True index
            results["n"] = t-1
            results["Accuracy at n"] = float(df.loc[t-1])
            results["Accuracy at n+1"] = float(df.loc[t])
        else:
            # Case where threshold is never met
            results["n"] = 1000
            results["Accuracy at n"] = float(df.iloc[-1])
            results["Accuracy at n+1"] = float(df.iloc[-1])
   
        if verbose is True:
            if mask.any():
                print(f"Optimal components by marginal gain: {t-1}")
            else:
                print(f"Marginal gain threshold unmet with all {df.shape[1]}components.")

        # Convert results to DataFrame all at once
        return results
