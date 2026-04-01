from random import random
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import curve_fit

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def calc_eval_stats(true_values, predicted, include_zeros = False):
    r2_list = []
    mse_list = []
    rmse_list = []
    mae_list = []
    all_stats = {}
    predicted = pd.DataFrame(predicted)
    true_values = pd.DataFrame(true_values)
    
    for i in range(true_values.shape[1]):
        x = true_values.iloc[:, i]
        y = predicted.iloc[:, i]
        
        # Check if column is all zeros
        if (x == 0).all():
            r2_list.append(9999)
            mse_list.append(9999)
            rmse_list.append(9999)
            mae_list.append(9999)
            continue
        
        if not include_zeros:
            x = x[x != 0]
            y = y.loc[x.index]
        
        r2 = r2_score(x, y)
        mse = mean_squared_error(x, y)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(x, y)
        
        r2_list.append(r2)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)

        
    all_stats["R2"] = r2_list
    all_stats["MSE"] = mse_list
    all_stats["RMSE"] = rmse_list
    all_stats['MAE'] = mae_list
    
    return all_stats

def fit_best_curve(x, y, criterion = ("R2", "max"), include_zeros = True):
        """Fit a set of lines to the data and decide on the best one based on R squared value.
        
        Args
            x (array-like): Independent variable data.
            y (array-like): Dependent variable data.
            criterion (str, str): Tuple of the criterion to evaluate the best fit on and how to choose best fit (min or max criterion value). Default is "R2". Options are MAE, RMSE, MSE, MAPE, R2.
            
        Output
            best_fit_params (tuple): Parameters of the best fit curve.
            all_curve_params (dict): Parameters of all fitted curves.
            best_fit_line (str) : Type of curve that fits best
            all_stats (dict) : Evaluation stats of the fitted curves.
        """
        
        all_curve_params = {}
        crit_ordering = criterion[1]
        criterion = criterion[0]

        if not crit_ordering.isin(["R2", "MAE"]):
            raise ValueError("Criterion ordering must be either 'max' or 'min'")
        
        try:
            p0 = [max(y), np.mean(x), 1, 0] # mandatory initial guess
            popt, _ = curve_fit(sigmoid, x, y, p0, method='trf', nan_policy="omit")
            y_pred_sigmoid = sigmoid(x, *popt)
            all_curve_params = {'sigmoid': popt}

            sigmoid_stats = calc_eval_stats(y, y_pred_sigmoid, include_zeros= include_zeros)
            all_stats = {'sigmoid' : sigmoid_stats}

        except:
            print("Sigmoid fit failed, defaulting to linear.")
            all_curve_params = {'sigmoid': None}
            all_stats = {'sigmoid' : None}

    
        # Fit straight line
        b, m = np.polyfit(x, y, 1)
        all_curve_params["linear"] = (b, m)
        y_pred_linear = b*x +m

        linear_stats = calc_eval_stats(y, y_pred_linear, include_zeros = include_zeros)
        all_stats['linear'] = linear_stats
        
        if all_stats['sigmoid'] is None:
            best_fit_line = "linear"
        else:
            # Pick best curve
            sigmoid_crit = sigmoid_stats[criterion][0]
            linear_crit = linear_stats[criterion][0]
            d = {"sigmoid": sigmoid_crit, "linear" : linear_crit}
            if crit_ordering == "max":
                best_fit_line = max(d, key = d.get)
            else:
                best_fit_line = min(d, key = d.get)
    
        best_fit_params = all_curve_params.get(best_fit_line)
        
        return best_fit_params, all_curve_params, best_fit_line, all_stats

def inverse_sigmoid(y, L, x0, k, b):
    """
    Inverse of sigmoid function.
    Given y, returns x.
    """
    y = np.asarray(y, dtype=float)

    # Check if y is in valid range
    bad = (y - b <= 0) | (y - b >= L)
    if np.any(bad):
        print("Warning: y values outside valid range (b < y < L+b)")

    # Small epsilon to avoid log(0) or negative arguments
    eps = 1e-12

    # Compute the valid range for y
    y_min = b + eps
    y_max = L + b - eps

    # Clip y to the range
    y = np.clip(y, y_min, y_max)
        
    x = x0 - (1/k) * np.log(L/(y - b) - 1)

    return x

class UnmixPixels:
    
    def __init__(self, x_train, y_train, x_test, y_test, 
                 presence_absence_thresh = 0.5, reload = False, reload_dict = None):
        self.x_train = x_train.copy()
        self.y_train = y_train.copy().set_index(x_train.index)
        self.y_train[np.isnan(self.y_train)] = 0
        self.x_test = x_test.copy()
        self.y_test = y_test.copy().set_index(x_test.index)
        self.y_test[np.isnan(self.y_test)] = 0
        self.presence_thresh = presence_absence_thresh
        self.water_regressor = None
        self.presence_probas = None
        if reload:
            try :
                self.classifiers = joblib.load(reload_dict["classifiers"])
            except:
                self.classifiers = None
            try:
                self.regressors = joblib.load(reload_dict["regressors"])
            except:
                self.regressors = None
            try: 
                self.water_regressor = joblib.load(reload_dict["water_regressor"])
            except : 
                self.water_regressor = None
        else:
            self.classifiers = None
            self.regressors = None
            self.water_regressor = None

    
    def classify_presence(self, classifier_params, plot, regressor_params):
        presence_probabilities = {
            "test": {},
            "train": {}
        }
        water_prediction = {
            "test": {},
            "train": {}
        }
        
        # WATER REGRESSION
        
        if self.water_regressor is None:      
            # Predict water fraction and add to features
            water_rfr = RandomForestRegressor(**regressor_params).fit(self.x_train, self.y_train.iloc[:, -1])
            self.water_regressor = water_rfr
            
        water_prediction["train"] = self.water_regressor.predict(self.x_train)
        water_prediction["test"] = self.water_regressor.predict(self.x_test)

        # Add water predictions to classifier input
        self.x_train["water"] = water_prediction["train"]
        self.x_train.columns = self.x_train.columns.astype(str)
        self.x_test["water"] = water_prediction["test"]
        self.x_test.columns = self.x_test.columns.astype(str)
        
        # Optional plotting
        if plot.lower() in ("water", "both"):
            plt.scatter(x = self.y_test.iloc[:, -1], y = water_prediction["test"], marker = 'o', alpha = 0.4, c= "teal")
            # plt.title("Regression predicted water FPC")
            plt.xlabel("True water FPC")
            plt.ylabel("Regression predicted water FPC")
            plt.axline((0,0), slope = 1, color = "black", linestyle = "--")
            plt.show()
        
        # CLASS-WISE PRESENCE/ABSENCE CLASSIFICATION
        if self.classifiers is None:
            # Store classifiers
            self.classifiers = {}
            
            # Train classifier for each column except water
            for col in self.y_train.columns[:-1]:
                
                #  Fit classifier to each column by presence
                rfc = RandomForestClassifier(**classifier_params).fit(self.x_train, self.y_train.loc[:,col]> 0)
                self.classifiers[col] = rfc
                
            
        for col in self.y_train.columns[:-1]:
            # Predict on both datasets
            test_probas = self.classifiers[col].predict_proba(self.x_test) 
            train_probas = self.classifiers[col].predict_proba(self.x_train) 
            
            print(f"{col.capitalize()} hard classification score: {self.classifiers[col].score(self.x_test, self.y_test.loc[:, col]>0)}")
            
            presence_probabilities["train"][col] = train_probas[:, 1]
            presence_probabilities["test"][col] = test_probas[:, 1]
        
        self.presence_probas = presence_probabilities

        return presence_probabilities, water_prediction
                
    def filter_and_regress(self, plot, regressor_params):
        
        class_predictions = {
            "train" : {},
            "test" : {}
        }
        
        # filter for present pixels by threshold
        test_presence_mask = pd.DataFrame.from_dict(self.presence_probas["test"]).set_index(self.y_test.index) > self.presence_thresh
        train_presence_mask = pd.DataFrame.from_dict(self.presence_probas["train"]).set_index(self.y_train.index) > self.presence_thresh
        
        if self.regressors is None: 
            self.regressors = {}
            for col in test_presence_mask.columns:
                # Find indices of pixels with predicted class presence
                present_test_idx  = self.x_test.index[test_presence_mask[col]]
                present_train_idx  = self.x_train.index[train_presence_mask[col]]

                #Train regressor only on present pixels
                reg = RandomForestRegressor(**regressor_params).fit(self.x_train.loc[present_train_idx, :], self.y_train.loc[present_train_idx, col])
                self.regressors[col] = reg
        
        for col in test_presence_mask.columns:
            # Regress results for all pixels
            test_predictions = pd.Series(self.regressors[col].predict(self.x_test), index = self.x_test.index)
            train_predictions = pd.Series(self.regressors[col].predict(self.x_train), index = self.x_train.index)
            
            class_predictions["train"][col] = pd.Series(train_predictions)
            class_predictions["test"][col] = pd.Series(test_predictions)

            if plot.lower() in ("regressors", "both"):
                #Predict on only present pixels
                plt.scatter(x = self.y_test.loc[:, col], 
                            y = test_predictions, marker = 'x', c= "grey", label = "All simulated pixels")
                plt.legend()
                plt.xlabel(f"True {col} FPC")
                plt.ylabel(f"Regression Predicted {col} FPC")       
                plt.show()

        return class_predictions

    def check_sensitivity(self, regression_predictions):
        """ Check how sensitive the regressors are to specific classes by investigating the results produced when single-class pixels (plus water) are run through a model trained on all classes. Perfectly sensitivity would mean that only the simulated class would produce regressed predictions.
        
        Args
        regression_predictions(Dataframe) : df of the predictions output by the regressors
        """
        # Make list of pixels with only one class plus water
        two_classes = self.y_test.loc[(self.y_test > 0 ).sum(axis = 1) ==2 ,:]

        # make single-class pixel index lists
        class_indices = {k: [] for k in self.y_test.columns[:-1]}
        for k, _ in class_indices.items():
            class_indices[k] = two_classes[two_classes[k] >0].index
            
        # Filter to single class pixels, fpcs, and predictions
        single_class_fpcs = {}
        single_class_predictions = {}

        for k , v in class_indices.items():
            single_class_fpcs[k] = self.y_test.loc[v, :]
            single_class_predictions[k] = regression_predictions.loc[v, :]

        # for each class, calculate the mean predicted FPC for the other classes store in df
        mean_FP = pd.DataFrame(index = class_indices.keys(), columns = class_indices.keys())
        for k in class_indices.keys():
            for k2 in class_indices.keys():
                if k != k2:
                    mean_FP.loc[k, k2] = single_class_predictions[k].loc[:, k2].mean()
        # For each class, calc the stdev of the predicted FPC for the other classes and store in other df
        std_FP = pd.DataFrame(index = class_indices.keys(), columns = class_indices.keys())
        for k in class_indices.keys():
            for k2 in class_indices.keys():
                if k != k2:
                    std_FP.loc[k, k2] = single_class_predictions[k].loc[:, k2].std()
        # Calc the mean FP value plus 1 std and plus 2 std

        mean_std_FP = (mean_FP + std_FP).astype(float).round(3)
        mean_std_FP.loc["average", :] = mean_std_FP.mean(axis = 0)
        mean_FP.loc["average", :] = mean_FP.mean(axis = 0)
        std_FP.loc["average", :] = std_FP.mean(axis = 0)


        # Calculate total pixels simulated of single classes plus water
        total_pixels_simmed = {k: ((single_class_fpcs[k][k] != 0).sum()) for k in single_class_fpcs.keys()}

        # summarize for each combo : number of FPs, number of TPs below thresh, sum of FP FPC, sum of TP FPC disregarded at that thresh
        counts_FP = pd.DataFrame(index = class_indices.keys(), columns = class_indices.keys())
        for k in class_indices.keys():
            for k2 in class_indices.keys():
                if k != k2:
                    counts_FP.loc[k, k2] = (single_class_predictions[k].loc[:, k2] != 0).sum()
                else:
                    counts_FP.loc[k, k] = total_pixels_simmed[k]

        return counts_FP, mean_FP, std_FP, mean_std_FP

    def apply_sensitivity(self, predictions, sensitivity_thresholds, plot = False): 
        """ Apply chosen sensitivity thresholding to predictions to get rid of small magnitude false positives"""

        # all values of test_regression below threshold go to 0

        test_results_threshed = predictions.copy()
        for col in test_results_threshed.columns[:-1]:
            test_results_threshed.loc[test_results_threshed[col] < sensitivity_thresholds[col], col] = 0
            
            if plot: 
                plt.scatter(x = self.y_test.loc[:, col], y = predictions.loc[:, col], alpha = 0.3, c = "red")
                plt.scatter(x = self.y_test.loc[:, col], y = test_results_threshed.loc[:, col], alpha = 0.3)
                plt.annotate(col, xy = (0.05, 0.85), xycoords= "axes fraction")
                plt.show()
        return test_results_threshed
       
    def plot_sensitivity_no_water(self, predicted_values, sensi_col, spectra : pd.DataFrame, possible_colours : list, endmembers : pd.DataFrame,FP_thresh_mode = "share", FP_thresh = None, ax_height = 2.5,  true_y = None,  savefig = False, plot_output_dir = None, cols_to_plot = None):
    
        """
        
        Args:
            Predicted_values (df): current predicted FPC values for all pixels and classes
            sensi_col: column (class) to plot as truly simulate to evaluate false positives of other classes
            FP_thresh_mode(str) : either count (percent of simulated pixels producing false positives) or share (percent of summed false positive FPC attributable to one endmember)
            FP_thresh :value to use, if none provided defaults to 5% for count and 2*sum(FP_FPC)/n where n is the number of unique contirbuting endmembers for share
            spectra (DF): endmember spectra
            possible_colours (list): list of colours that can be used for plotting
        """
    
        # Initialize variables
        if true_y is None:
            true_y = self.y_test

        test_endmember_indices = endmembers[true_y.index, :]
            
        single_class_fpcs = {}
        single_class_predictions = {}
        contributing_endmember_indices = {}

        # make list of pixels with only two classes
        two_classes = true_y.loc[(true_y > 0 ).sum(axis = 1) ==2 ,:]

        # List all pixels with only the COI simulated
        class_indices = {k: two_classes[two_classes[k] >0].index for k in true_y.columns[:-1]}

        for k , v in class_indices.items():
            single_class_fpcs[k] = true_y.loc[v, :]
            single_class_predictions[k] = predicted_values.loc[v, :]
            contributing_endmember_indices[k] = pd.DataFrame(test_endmember_indices, index = true_y.index, columns= true_y.columns).loc[v, k]

        # Choose FP level threshold: percent of whole by pixels simulated; share : percent of all FP detections
        if FP_thresh_mode == "share":
            if FP_thresh is None:
                FP_share= 100/len(contributing_endmember_indices[sensi_col].unique()) * 2
            else: 
                FP_share = 100/len(contributing_endmember_indices[sensi_col].unique()) * FP_thresh
            print(f"Thresholding by false positive share.\nEndmembers producing more than {round(FP_share, 1)} % of false positives will be identified.")
        elif FP_thresh_mode == "count":
            if FP_thresh is None:
                FP_thresh = 0.006
            # How many false positives is considered too many?
            b = int(len(single_class_fpcs[sensi_col]) * FP_thresh)
            print(f"Thresholding by pixel count. \n{FP_thresh *100} % false positive threshold: {b} pixels")

        not_simulated_classes = single_class_predictions[sensi_col].drop([sensi_col, "water"], axis = 1)
        single_class_predictions[sensi_col]["contributor"] = contributing_endmember_indices[sensi_col]
        single_class_predictions[sensi_col]["true_sensi_col"] = single_class_fpcs[sensi_col][sensi_col]
        single_class_predictions[sensi_col]["true_water"] = single_class_fpcs[sensi_col]["water"]
        
        # Make dictionary of colours to plot the endmembers as
        all_band_colours = {i[1]: possible_colours[i[0]] for i in enumerate(contributing_endmember_indices[sensi_col].unique())}

        # Plot sensitivity with only one not-simulated class
        if not_simulated_classes.shape[1] == 1: 
            fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (9, ax_height+0.5), layout = "constrained")
            
            #cycle through rows in the plots and columns in the data
            for i in enumerate(not_simulated_classes):
                endmem_colours = all_band_colours.copy()
                sensitive_data = single_class_predictions[sensi_col]
                
                # Find all rows with false positives in this class
                false_positives = sensitive_data.loc[:, i[1]] > 0

                #Count FPs by contributing endmember
                false_posi_counts = sensitive_data.loc[false_positives, "contributor"].value_counts()

                # Find endmembers making more FP detections than limit (either number or share)
                if FP_thresh_mode == "share":
                    contributor_shares = single_class_predictions[sensi_col].pivot_table(values = i[1], columns = "contributor", aggfunc = "sum")/single_class_predictions[sensi_col][i[1]].sum() *100
                    outlying_bands = contributor_shares.columns[(contributor_shares> FP_share).all()].values
                    for x in contributing_endmember_indices[sensi_col].unique():
                        if x not in outlying_bands:
                            endmem_colours[x] = "grey"
                elif FP_thresh_mode == "count":
                    outlying_bands = [int(x) for x in false_posi_counts[false_posi_counts > b].index]
                    endmem_colours = {endmem_colours[x] : "grey" for x in outlying_bands}
                else:
                    raise ValueError("FP thresholding mode must be either 'share' or 'count'.")
                
                # Add true simulated values, endmember, colour/alpha allocations
                sensitive_data["colour"] = [endmem_colours.get(x, "grey") for x in sensitive_data["contributor"]]
                sensitive_data["alpha"] = [0.2 if x == "grey" else 0.9 for x in sensitive_data["colour"]]

                # Split indices for this row
                grey_indices = [x for x in sensitive_data[sensitive_data["colour"] == "grey"]["contributor"].unique()]
                coloured_indices = [x for x in sensitive_data[sensitive_data["colour"] != "grey"]["contributor"].unique()]
                
                ordered_indices = grey_indices+coloured_indices
                
                for idx in enumerate(ordered_indices):

                    single_endmem_data = sensitive_data[sensitive_data["contributor"] == idx[1]]
                    
                    # Use this for subplot col 2 if you don't want to plot 0s
                    #single_endmem_FP = single_endmem_data.loc[false_positives, :]
                
                    ax[0].scatter(
                                x = single_endmem_data["true_sensi_col"],
                                y = single_endmem_data[sensi_col],
                                color = single_endmem_data["colour"],
                                label = idx,
                                alpha = single_endmem_data["alpha"],
                                linewidths = 0,
                                s = 15
                                )
                    ax[0].set_ylabel("Predicted FPC")                   
                    ax[1].scatter(
                                x=[idx[0]+random()*0.05 for x in range(single_endmem_data.shape[0])],
                                y=single_endmem_data[i[1]],
                                c=single_endmem_data["colour"],
                                s=8,
                                linewidths=0
                                )
                    ax[1].hlines(sensitive_data[i[1]].mean(),
                                xmin = 0,
                                xmax = len(ordered_indices),
                                colors = "blue",
                                linewidth = 0.4)
                    ax[1].hlines(sensitive_data[i[1]].mean()+2*sensitive_data[i[1]].std(),
                                    xmin = 0,
                                    xmax = len(ordered_indices),
                                    colors = "red",
                                    linestyles = "--",
                                    linewidth = 0.6
                                    )
                    ax[1].hlines(sensitive_data[i[1]].mean()+sensitive_data[i[1]].std(),
                                    xmin = 0,
                                    xmax = len(ordered_indices),
                                    colors = "black",
                                    linestyles = ":",
                                    linewidth = 0.6)
                    
                    ax[1].set_xticks([])
                    ax[1].annotate(i[1].capitalize().replace("_", " "), xy = (0.2, 0.6), xycoords = "axes fraction", size = 12)
                    
                    ax[2].plot(spectra.loc[idx[1], :].T,
                            color = endmem_colours.get(idx[1], "grey"),
                            label = idx[1],
                            alpha = 0.25 if endmem_colours.get(idx[1], "grey")== "grey" else 1)
                    ax[2].set_ylabel("Reflectance")
                    ax[2].set_xticks([])

            ax[0].set_title(f"Predicted {sensi_col.lower().replace("_", " ")} FPC")
            ax[0].annotate("A", xy = (0.05, 0.8), xycoords = "axes fraction", size = 16)
            ax[1].set_title("False positive FPC predictions\nby simulation endmember")
            ax[1].annotate("B", xy = (0.05, 0.8), xycoords = "axes fraction", size = 16)
            ax[2].set_title("Contributing endmembers'\nspectral profiles")
            ax[2].annotate("D", xy = (0.05, 0.8), xycoords = "axes fraction", size = 16)
            ax[0].set_xlabel(f"True simulated {sensi_col.capitalize().replace('_', ' ')} FPC")
            ax[1].set_xlabel("False positive FPC predictions\nby endmember")
            if len(spectra.columns) < 12:
                ax[2].set_xticks(spectra.columns)
                ax[2].set_xlabel("Sensor band")
            else:
                ax[2].set_xticks(spectra.columns[np.arange(0, len(spectra.columns), 10)])
                ax[2].set_xticklabels(spectra.columns[np.arange(0, len(spectra.columns), 10)])
                ax[2].set_xlabel("Sensor band (nm)")
            
        else:
            if cols_to_plot is not None:
                not_simulated_classes = not_simulated_classes.loc[:, cols_to_plot]
            fig, ax = plt.subplots(not_simulated_classes.shape[1], 3, figsize = (9, not_simulated_classes.shape[1]*ax_height), sharey = "col", layout = "constrained" )
        

            #cycle through rows in the plots and columns in the data
            for i in enumerate(not_simulated_classes):
                endmem_colours = all_band_colours.copy()
                sensitive_data = single_class_predictions[sensi_col]
                
                # Find all rows with false positives in this class
                false_positives = sensitive_data.loc[:, i[1]] > 0

                #Count FPs by contributing endmember
                false_posi_counts = sensitive_data.loc[false_positives, "contributor"].value_counts()

                # Find endmembers making more FP detections than limit (either number or share)
                if FP_thresh_mode == "share":
                    contributor_shares = single_class_predictions[sensi_col].pivot_table(values = i[1], columns = "contributor", aggfunc = "sum")/single_class_predictions[sensi_col][i[1]].sum() *100
                    outlying_bands = contributor_shares.columns[(contributor_shares> FP_share).all()].values
                    for x in contributing_endmember_indices[sensi_col].unique():
                        if x not in outlying_bands:
                            endmem_colours[x] = "grey"
                elif FP_thresh_mode == "count":
                    outlying_bands = [int(x) for x in false_posi_counts[false_posi_counts > b].index]
                    endmem_colours = {endmem_colours[x] : "grey" for x in outlying_bands}
                else:
                    raise ValueError("FP thresholding mode must be either 'share' or 'count'.")
                
                # Add true simulated values, endmember, colour/alpha allocations
                sensitive_data["colour"] = [endmem_colours.get(x, "grey") for x in sensitive_data["contributor"]]
                sensitive_data["alpha"] = [0.2 if x == "grey" else 0.9 for x in sensitive_data["colour"]]

                # Split indices for this row
                grey_indices = [x for x in sensitive_data[sensitive_data["colour"] == "grey"]["contributor"].unique()]
                coloured_indices = [x for x in sensitive_data[sensitive_data["colour"] != "grey"]["contributor"].unique()]
                
                ordered_indices = grey_indices+coloured_indices
                
                for idx in enumerate(ordered_indices): 

                    single_endmem_data = sensitive_data[sensitive_data["contributor"] == idx[1]]
                    
                    # Use this for subplot col 2 if you don't want to plot 0s
                    #single_endmem_FP = single_endmem_data.loc[false_positives, :]
                
                    ax[i[0], 0].scatter(
                                x = single_endmem_data["true_sensi_col"],                     
                                y = single_endmem_data[sensi_col], 
                                color = single_endmem_data["colour"], 
                                label = idx, 
                                alpha = single_endmem_data["alpha"],
                                linewidths = 0,
                                s = 15
                                )
                    ax[i[0], 0].set_ylabel("Predicted FPC")
                    ax[i[0], 1].scatter(
                                    x=[idx[0]+random()*0.08 for x in range(single_endmem_data.shape[0])],
                                    y=single_endmem_data[i[1]],
                                    c=single_endmem_data["colour"],
                                    s=8,
                                    linewidths=0
                                    )
                    ax[i[0], 1].hlines(sensitive_data[i[1]].mean(), 
                                    xmin = 0, 
                                    xmax = len(ordered_indices), 
                                    colors = "blue", 
                                    linewidth = 0.4)
                    ax[i[0], 1].hlines(sensitive_data[i[1]].mean()+2*sensitive_data[i[1]].std(), 
                                    xmin = 0, 
                                    xmax = len(ordered_indices), 
                                    colors = "red", 
                                    linestyles = "--", 
                                    linewidth = 0.6
                                    )
                    ax[i[0], 1].hlines(sensitive_data[i[1]].mean()+sensitive_data[i[1]].std(), 
                                    xmin = 0, 
                                    xmax = len(ordered_indices), 
                                    colors = "black", 
                                    linestyles = ":", 
                                    linewidth = 0.6)
                    
                    ax[i[0], 1].set_xticks([])
                    ax[i[0], 1].annotate(i[1].capitalize().replace("_", " "), xy = (0.2, 0.7), xycoords = "axes fraction", size = 10)
                    
                    ax[i[0], 2].plot(spectra.loc[idx[1], :].T, 
                            color = endmem_colours.get(idx[1], "grey"), 
                            label = idx[1],
                            alpha = 0.25 if endmem_colours.get(idx[1], "grey") == "grey" else 1)
                    ax[i[0], 2].set_ylabel(f"Reflectance")
                    if len(spectra.columns) < 12:
                        ax[i[0], 2].set_xticks(spectra.columns)
                        ax[i[0], 2].set_xticklabels([])
                    else:
                        ax[i[0], 2].set_xticks(spectra.columns[np.arange(0, len(spectra.columns), 10)])
                        ax[i[0], 2].set_xticklabels([])

            ax[0,0].set_title(f"Predicted {sensi_col.lower().replace("_", " ")} FPC")
            ax[0,0].annotate("A", xy = (0.05, 0.8), xycoords = "axes fraction", size = 16)
            ax[0,1].set_title("False positive FPC predictions\nby simulation endmember")
            ax[0,1].annotate("B", xy = (0.05, 0.8), xycoords = "axes fraction", size = 16)
            ax[0,2].set_title("Contrbuting endmembers'\nspectral profiles")
            ax[0,2].annotate("C", xy = (0.05, 0.8), xycoords = "axes fraction", size = 16)
            ax[-1, 0].set_xlabel(f"True simulated {sensi_col.lower().replace("_", " ")} FPC")
            ax[-1, 1].set_xlabel("False positive FPC predictions\nby endmember")
            if len(spectra.columns) < 12:
                ax[-1, 2].set_xticks(spectra.columns)
                ax[-1, 2].set_xticklabels(spectra.columns)
                ax[-1, 2].set_xlabel("Sensor band")
            else:
                ax[-1, 2].set_xticks(spectra.columns[np.arange(0, len(spectra.columns), 10)])
                ax[-1, 2].set_xticklabels(spectra.columns[np.arange(0, len(spectra.columns), 10)])
                ax[-1, 2].set_xlabel("Sensor band (nm)")
            
            
        plt.xticks(rotation = 45)
        if savefig:
            plt.savefig(f"{plot_output_dir}false_positive_endmember_investigation_NOWATER.svg", bbox_inches = "tight")
            plt.savefig(f"{plot_output_dir}false_positive_endmember_investigation_NOWATER.png", bbox_inches = "tight")
        plt.show()
        
        return
        
    def fit_all_curves(self, predicted_values, criterion = ("R2", "max"), include_zeros = False):
        fitted_curves = {}
        all_stats = {}
        all_curves = {}

        for col in self.y_test.columns:
            cleaned_predictions = predicted_values[predicted_values[col] != 0][col]
            best_fit_params, all_curve_params, best_fit_line, curve_stats = fit_best_curve(self.y_test.loc[cleaned_predictions.index, col], 
                cleaned_predictions,
                include_zeros = include_zeros)
            
            print(f"Best fit curve is a {best_fit_line.capitalize()} for {col.replace('_', ' ' )} with {criterion[0]} = {round(curve_stats[best_fit_line][criterion[0]][0], 4)}")
            fitted_curves[col] = best_fit_params
            all_stats[col] = curve_stats
            all_curves[col] =  all_curve_params
            
        self.fitted_curves = fitted_curves
        self.all_curves = all_curves
        
        return fitted_curves, all_stats, all_curves
    
    def apply_inversion(self, data_to_invert = pd.DataFrame, force_fit = None):
        """ 
        Invert the best fitted curves to back-calculate the expected/adjusted FPC values for any given regression output.
        
        """
        predicted_covers = pd.DataFrame(index = data_to_invert.index, columns = data_to_invert.columns)
        
        for col in data_to_invert.columns: #[:-1]:
            
            # Check if line fit choice is forced
            if force_fit is not None:
                if force_fit not in ["sigmoid", "linear"]:
                    raise ValueError("Forced fit not recognised.")
                else:
                    chosen_fit = force_fit
            # If not, pick line fit based on best fit
            else:
                if len(self.fitted_curves[col]) == 4:
                    chosen_fit = "sigmoid"
                elif len(self.fitted_curves[col]) == 2: 
                    chosen_fit = "linear"
                else:
                    raise ValueError("Fitted curve parameters unexpected size.")
            
            if chosen_fit == "sigmoid": 
                # Access sigmoid params
                L, x0, k, b = self.all_curves[col][chosen_fit]

                # Calculate inverse sigmoid for predicted values
                x_result = inverse_sigmoid(data_to_invert.loc[:, col], L, x0, k, b)
                x_result = np.clip(x_result, 0, 1)
                
                # Store new predicted values
                predicted_covers[col] = x_result
                
            elif chosen_fit == "linear":
                # Access linear params
                b, m = self.all_curves[col][chosen_fit]
                
                # Calculate inverse linear for predicted values
                x_result = (data_to_invert.loc[:, col] - m) / b
                x_result = np.clip(x_result, 0, 1)
                
                # Store new predicted values
                predicted_covers[col] = x_result
            
        return predicted_covers
      
    def unmix_new_data(self, new_pixels, fpcs_columns, sensitivity_thresholds = None):
        
        # Format input pixels
        new_pixels = new_pixels.copy()
        class_probas = pd.DataFrame(index = new_pixels.index, columns = fpcs_columns[:-1])
        class_regressions = pd.DataFrame(index = new_pixels.index, columns = fpcs_columns[:-1])
        new_pixels.columns = new_pixels.columns.astype(str)
        
        # Predict water FPC and add to input features
        new_pixels["water"] = self.water_regressor.predict(new_pixels)
        
        # Predict presence/absence and add to features
        for col, classifier in self.classifiers.items():
            new_probas = classifier.predict_proba(new_pixels) 
            class_probas[col] = new_probas[:, 1]
            
        # Predict fractional abundances
        for col, regressor in self.regressors.items():
            class_regressions[col] = regressor.predict(new_pixels)
        
        if sensitivity_thresholds is not None:
            class_regressions = self.apply_sensitivity(class_regressions, sensitivity_thresholds)
                     
        return class_probas, new_pixels["water"], class_regressions
    