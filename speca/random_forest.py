import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score,balanced_accuracy_score

def perform_rfc(x_train, x_test, y_train, y_test, classes, hyperparameters = None):
        """ Do RF classification"""

        # Make results placeholder
        results = {}
        
        # Initialize Randon Forest with hyperparameters if given
        rf = RFC(hyperparameters)
        rf.fit(x_train, y_train)
        predictions = rf.predict(x_test)
        results["accuracy"] = rf.score(x_test, y_test)
        results["matrix"] = confusion_matrix(y_test, predictions, labels= classes, normalize = 'true')
        results["recall"] = recall_score(y_test, predictions, average = "weighted")
        results["precision"] = precision_score(y_test, predictions, average = "weighted", zero_division=np.nan)
        results["bal_acc"] = balanced_accuracy_score(y_test, predictions) 
        return results
        

class RFC:
    """
    Class to parameterize and run many random forests with sample re-division then plot results.
    """
    def __init__(self,
                 spectra,
                 labels,
                 n_comps,
                 labels_col = "Class",
                 runs = 100):
        """
        Initialize the random forest classifier class.
        Args:
            spectra (pd.DataFrame): Reflectance data.
            labels (pd.DataFrame): Labels for the spectra.
        """
        self.spectra = spectra
        self.labels = labels
        self.labels_col = labels_col
        self.n_comps = n_comps
        self.runs = runs
        
        # Placeholders for later
        self.rfc_results = None


    def grid_search(self):
        
        grid = RFC.grid_search()
        hyperparams = {}
        self.hyperparams = hyperparams


    def many_rfc_runs(self):
        """"
        Aggregate the various scores of many runs of the random forest classifier
        """
        dep = self.labels[self.labels_col]
        
        # Create placeholder variables
        accuracies = []
        recalls = []
        precisions = []
        bal_accs = []
        cm = np.zeros(shape = (len(dep.unique()), len(dep.unique())))
        cumsums = np.zeros(self.n_comps)
        
        for i in range(self.runs):
            x_train, x_test, y_train, y_test = train_test_split(self.spectra, self.labels[self.labels_col], test_size= 0.3, stratify = self.labels[self.labels_col]) 
            run= perform_rfc(x_train, x_test, y_train, y_test, hyperparameters=self.hyperparams, classes = dep.unique())
    
            accuracies.append(run["accuracy"])
            recalls.append(run["recall"])       
            precisions.append(run["precision"]) 
            bal_accs.append(run["bal_acc"])     		
            cm += run["matrix"]
            cumsums += run["variance cum sum"]
    
        print(f"Mean accuracy for {self.labels_col} is {np.mean(accuracies)}.")
        print(f"Mean recall for {self.labels_col} is {np.mean(recalls)}.")
        print(f"Mean precision for {self.labels_col} is {np.mean(precisions)}.")
        print(f"Mean balanced accuracy for {self.labels_col} is {np.mean(bal_accs)}.")
        
        self.rfc_results = {
            "accuracies" : accuracies,
            "recalls" : recalls,
            "precisions" : precisions,
            "balanced accuracies" : bal_accs,
            "cm" : cm,
            "variance cumsum" : cumsums
        }


    def plot_cm(self):
        """"
        Plot the confusion matrix of the Random Forest classifier.
        """
        # Make confusion matrix object into dataframe
        cm_df = pd.DataFrame(self.rfc_results/self.runs)
        
        # Mask values that round to 0.00 for visula clarity
        cm_masked = cm_df.map(lambda v: str(round(v, 2)) if v > 0 else "")
        
        # Make class labels list
        dep = self.labels[self.labels_col]
        #TODO: Adjust for class-site classifications not just class.
         
        # Plot the confusion matrix
        plt.close()
        sns.heatmap(self.rfc_results["cm"]/self.runs,
                    annot = cm_masked,
                    fmt = "s",
                    cmap ='Blues',
                    xticklabels = dep.unique(),
                    yticklabels = dep.unique(),
                    annot_kws = {"size" : 6},
                    vmin = 0,
                    vmax = 1
                    )
        plt.tight_layout(pad = 4, w_pad = 1, h_pad = 1)
        plt.xlabel('Predicted Labels')
        plt.xticks(rotation = 90)
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix for {self.labels_col} - RFC')
        plt.show()
        
    def plot_decision_space(self, comp_1, comp_2):
        """
        Plot the decision space of two components using the full dataset
        """

        feature_1, feature_2 = np.meshgrid(
            np.linspace(comp_1.min(), comp_1.max()),
            np.linspace(comp_2.min(), comp_2.max()))

        grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

        forest = RandomForestClassifier(self.hyperparams).fit(self.spectra,
                                                              self.labels[self.labels_col])

        y_pred = np.reshape(forest.predict(grid), feature_1.shape)
        display = DecisionBoundaryDisplay(
            xx0=feature_1, xx1=feature_2, response=y_pred)
        display.plot()
        display.ax_.scatter(self.spectra[:, comp_1],
                            self.spectra[:, comp_2],
                            c=self.labels[self.labels_col],
                            edgecolor="black")
        plt.show()