import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, precision_score,balanced_accuracy_score, f1_score

def perform_rfc(x_train, x_test, y_train, y_test, classes, hyperparameters = None):
    """ Do RF classification and store results."""

    # Make results placeholder
    results = {}
    
    # Initialize Randon Forest with hyperparameters if given
    rf = RandomForestClassifier(**hyperparameters)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    results["accuracy"] = rf.score(x_test, y_test)
    results["matrix"] = confusion_matrix(y_test, predictions, labels= classes, normalize = 'true')
    results["recall"] = recall_score(y_test, predictions, average = "weighted")
    results["precision"] = precision_score(y_test, predictions, average = "weighted", zero_division=np.nan)
    results["bal_acc"] = balanced_accuracy_score(y_test, predictions)
    results["f1"]= f1_score(y_test, predictions, average = "macro", zero_division=np.nan)
    results["importances"] = rf.feature_importances_
    return results


class RandoForest:
    """
    Class to parameterize and run many random forests with sample re-division then plot results.
    """
    def __init__(self,
                 spectra,
                 labels,
                 #labels_col = None,
                 runs = 100):
        """
        Initialize the random forest classifier class.
        Args:
            spectra (pd.DataFrame): Reflectance data as PCA components
            labels (pd.DataFrame): Labels for the spectra.
        """
        self.spectra = spectra
        self.labels = labels
        #self.labels_col = labels_col if labels_col is not None else "Class"
        self.runs = runs
        
        # Placeholders for later
        self.rfc_results = None
        self.hyperparams = None


    def grid_search(self, parameters = None):
        """ Performs a grid search of the parameter values defined below."""
        forest = RandomForestClassifier()
        params_dt_default = {'min_samples_split': [2, 3, 4],
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30],
                'max_leaf_nodes': [20, 40, 60, None]
                }
        
        # Set parameters to use if option is provided
        params_dt = parameters if parameters is not None else params_dt_default
        
        params_grid = GridSearchCV(estimator=forest,
                            param_grid=params_dt,
                            scoring='f1_macro',
                            cv=3,
                            n_jobs=-1)

        params_grid.fit(self.spectra, self.labels)  #[self.labels_col]
        print(f" Best parameters: {params_grid.best_params_}")
        print(f" Best score: {params_grid.best_score_}")
        self.hyperparams = params_grid.best_params_

    def many_rfc_runs(self):
        """"
        Aggregate the various scores of many runs of the random forest classifier
        """
        # if len(self.labels.shape) >1:
        #     dep = self.labels #[self.labels_col]
        # else:
        #     dep = self.labels

        dep = self.labels
        # Create placeholder variables
        accuracies = []
        recalls = []
        precisions = []
        bal_accs = []
        f1s = []
        cm = np.zeros(shape = (self.runs, len(dep.unique()), len(dep.unique())))
        importances = []
        
        # for i in range(self.runs):
        #     x_train, x_test, y_train, y_test = train_test_split(self.spectra,self.labels[self.labels_col], test_size= 0.3, stratify = self.labels[self.labels_col])
        #     run= perform_rfc(x_train, x_test, y_train, y_test,hyperparameters=self.hyperparams, classes = dep.unique())
    
        #     accuracies.append(run["accuracy"])
        #     recalls.append(run["recall"])
        #     precisions.append(run["precision"])
        #     bal_accs.append(run["bal_acc"])
        #     cm[i] = run["matrix"]
        #     f1s.append(run["f1"])
            
        for i in range(self.runs):
            x_train, x_test, y_train, y_test = train_test_split(self.spectra,dep, test_size= 0.3, stratify = dep)
            run= perform_rfc(x_train, x_test, y_train, y_test,hyperparameters=self.hyperparams, classes = dep.unique())
    
            accuracies.append(run["accuracy"])
            recalls.append(run["recall"])
            precisions.append(run["precision"])
            bal_accs.append(run["bal_acc"])
            cm[i] = run["matrix"]
            importances.append(run["importances"])
            
            f1s.append(run["f1"])
    
        print(f"Mean accuracy for {self.labels.name} is {np.mean(accuracies)}.") #{self.labels_col}
        print(f"Mean recall is {np.mean(recalls)}.") #{self.labels_col}
        print(f"Mean precision is {np.mean(precisions)}.") #for {self.labels_col}
        print(f"Mean balanced accuracy  is {np.mean(bal_accs)}.") #for {self.labels_col}
        print(f"Mean F1 score is {np.mean(f1s)}.") #for {self.labels_col} 
        
        self.rfc_results = {
            "accuracies" : accuracies,
            "recalls" : recalls,
            "precisions" : precisions,
            "balanced accuracies" : bal_accs,
            "cm" : cm,
            "f1_scores": f1s,
            "importances" : importances
        }


    def plot_cm(self):
        """"
        Plot the confusion matrix of the Random Forest classifier.
        """
        # Make confusion matrix object into dataframe
        cm_df = pd.DataFrame(np.mean(self.rfc_results["cm"], axis = 0))

        # Mask values that round to 0.00 for visula clarity
        cm_masked = cm_df.map(lambda v: str(int(v*100)) if int(v*100) > 0 else "")

        dep = self.labels

        # Plot the confusion matrix
        plt.close()
        plt.figure(figsize=(10, 8))
        sns.set_theme(font = "Times New Roman")
        sns.heatmap(np.mean(self.rfc_results["cm"], axis = 0)*100,
                    annot = cm_masked,
                    fmt = "s",
                    cmap ='Blues',
                    xticklabels = dep.unique(),
                    yticklabels = dep.unique(),
                    annot_kws = {"size" : 12},
                    vmin = 0,
                    vmax = 100
                    )
        plt.tight_layout(pad = 4, w_pad = 1, h_pad = 1)
        plt.xlabel('Predicted Labels')
        # plt.xticks(rotation = 30)
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix  - RFC')
        plt.savefig("./plots/RFC/RFC_cm.svg")
        plt.show()


    #TODO: finish plot decision space functionality
    # def plot_decision_space(self, comp_1, comp_2):
    #     """
    #     Plot the decision space of two components using the full dataset
    #     """

    #     feature_1, feature_2 = np.meshgrid(
    #         np.linspace(comp_1.min(), comp_1.max()),
    #         np.linspace(comp_2.min(), comp_2.max()))

    #     grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

    #     forest = RandomForestClassifier(self.hyperparams).fit(self.spectra,
    #                                                           self.labels[self.labels_col])

    #     y_pred = np.reshape(forest.predict(grid), feature_1.shape)
    #     display = DecisionBoundaryDisplay(
    #         xx0=feature_1, xx1=feature_2, response=y_pred)
    #     display.plot()
    #     display.ax_.scatter(self.spectra[:, comp_1],
    #                         self.spectra[:, comp_2],
    #                         c=self.labels[self.labels_col],
    #                         edgecolor="black")
    #     plt.show()
    