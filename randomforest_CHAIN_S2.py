import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor


class UnmixPixels:
    
    def __init__(self, x_train, y_train, x_test, y_test, 
                 presence_absence_thresh = 0.5):
        self.x_train = x_train.copy()
        self.y_train = y_train.copy().set_index(x_train.index)
        self.y_train[np.isnan(self.y_train)] = 0
        self.x_test = x_test.copy()
        self.y_test = y_test.copy().set_index(x_test.index)
        self.y_test[np.isnan(self.y_test)] = 0
        self.presence_thresh = presence_absence_thresh
        self.water_regressor = None
        self.classifiers = None
        self.regressors = None
        self.presence_probas = None
    
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
        
        # # Predict water fraction and add to features
        # water_rfr = RandomForestRegressor(**regressor_params).fit(self.x_train, self.y_train.iloc[:, -1])
        # water_prediction["train"] = water_rfr.predict(self.x_train)
        # water_prediction["test"] = water_rfr.predict(self.x_test)
        
        # Predict water fraction and add to features
        water_rfr = RandomForestRegressor(**regressor_params).fit(self.x_train, self.y_train.iloc[:, -1])
        water_prediction["train"] = water_rfr.predict(self.x_train)
        water_prediction["test"] = water_rfr.predict(self.x_test)

        # Store water regressor for unmixing new data
        self.water_regressor = water_rfr
        
        # Optional plotting
        if plot.lower() in ("water", "both"):
            plt.scatter(x = self.y_test.iloc[:, -1], y = water_prediction["test"], marker = 'o', alpha = 0.4, c= "teal")
            # plt.title("Regression predicted water FPC")
            plt.xlabel("True water FPC")
            plt.ylabel("Regression predicted water FPC")
            plt.axline((0,0), slope = 1, color = "black", linestyle = "--")
            plt.show()
        
        # CLASS-WISE PRESENCE/ABSENCE CLASSIFICATION
        
        # Store classifiers
        classifiers = {}
        
        # Train classifier for each column except water
        for col in self.y_train.columns[:-1]:
            
            # Add water predictions to classifier input
            self.x_train["water"] = water_prediction["train"]
            self.x_train.columns = self.x_train.columns.astype(str)
            self.x_test["water"] = water_prediction["test"]
            self.x_test.columns = self.x_test.columns.astype(str)
            
            #  Fit classifier to each column by presence
            rfc = RandomForestClassifier(**classifier_params).fit(self.x_train, self.y_train.loc[:,col]> 0)

            # Predict on both datasets
            test_probas = rfc.predict_proba(self.x_test) 
            train_probas = rfc.predict_proba(self.x_train) 
            
            print(f"{col.capitalize()} hard classification score: {rfc.score(self.x_test, self.y_test.loc[:, col]>0)}")
            
            # Store classification results and models
            # self.train_rfc[col] = train_probas[:,1] > self.presence_thresh
            # self.train_rfc[col] = test_probas[:, 1] > self.presence_thresh
            
            presence_probabilities["train"][col] = train_probas[:, 1]
            presence_probabilities["test"][col] = test_probas[:, 1]

            classifiers[col] = rfc
            
            # Optional plotting
            if plot.lower() in ("classifier", "both"):
                plt.scatter(x = self.y_test.loc[:, col], 
                            y = test_probas[:, 1], 
                            marker = 'o', 
                            alpha = 0.4, 
                            c = test_probas[:, 1] > self.presence_thresh, 
                            label = "Probability of Presence")
                plt.scatter(x = self.y_test.loc[:, col],
                            y = test_probas[:, 1]> 0.5,
                            alpha = 0.4,
                            c="red",
                            label = "Hard classification")
                plt.legend()
                plt.xlabel(f"True {col} FPC - test dataset")
                plt.ylabel(f"Predicted probability of {col} presence")
                plt.show()
           
            
        self.classifiers = classifiers 
        self.presence_probas = presence_probabilities
        # presence_probabilities["train"]["water"] = self.x_train["pred_water"]
        # presence_probabilities["test"]["water"] = self.x_test["pred_water"]

        return presence_probabilities, water_prediction
                

    def filter_and_regress(self, plot, regressor_params):
        
        class_predictions = {
            "train" : {},
            "test" : {}
        }
        regressors = {}
        
        # filter for present pixels by threshold
        test_presence_mask = pd.DataFrame.from_dict(self.presence_probas["test"]).set_index(self.y_test.index) > self.presence_thresh
        train_presence_mask = pd.DataFrame.from_dict(self.presence_probas["train"]).set_index(self.y_train.index) > self.presence_thresh
        
        for col in test_presence_mask.columns:
            # Find indices of pixels with predicted class presence
            present_test_idx  = self.x_test.index[test_presence_mask[col]]
            present_train_idx  = self.x_train.index[train_presence_mask[col]]
            print(f"{col} present in {sum(test_presence_mask[col])} out of {test_presence_mask.shape[0]} test pixels")
            print(f"Training set for {col} has {self.y_train.shape[0]} samples")
            
            
            #Train regressor only on present pixels
            reg = RandomForestRegressor(**regressor_params).fit(self.x_train.loc[present_train_idx, :], self.y_train.loc[present_train_idx, col])
            
            # #Train regressors on all pixels
            # reg = RandomForestRegressor(**regressor_params).fit(self.x_train, self.y_train[col])
            
            
            self.y_train[np.isnan(self.y_train)] = 0
            # Regress results for all pixels
            test_predictions = pd.Series(reg.predict(self.x_test), index = self.x_test.index)
            train_predictions = pd.Series(reg.predict(self.x_train), index = self.x_train.index)
            
            class_predictions["train"][col] = pd.Series(train_predictions)
            class_predictions["test"][col] = pd.Series(test_predictions)
            
            regressors[col] = reg
            
            
            # Optional plotting
            
            #Predict on only present pixels
            present_predictions_test = test_predictions[present_test_idx]
            #reg.predict(self.x_test.loc[present_test_idx, :])
            
            if plot.lower() in ("regressors", "both"):
                plt.scatter(x = self.y_test.loc[:, col], 
                            y = test_predictions, marker = 'x', c= "grey", label = "All simulated pixels")

                plt.scatter(x = self.y_test.loc[present_test_idx, col], 
                            y = present_predictions_test, marker = 'o', alpha = 0.4, c= self.x_test.loc[present_test_idx, "water"], label = "Pixels classified as class-present")
                plt.legend()
                plt.xlabel(f"True {col} FPC")
                plt.ylabel(f"Regression Predicted {col} FPC")       
                plt.show()

        self.regressors = regressors
        return class_predictions
    
    def unmix_new_data(self, new_pixels, fpcs_columns):
        
        # Format input pixels
        new_pixels = new_pixels.copy()
        class_probas = pd.DataFrame(index = new_pixels.index, columns = fpcs_columns[:-1])
        class_regressions = pd.DataFrame(index = new_pixels.index, columns = fpcs_columns[:-1])
        new_pixels.columns = new_pixels.columns.astype(str)
        
        # Predict water FPC and add to input features
        new_pixels["water"] = self.water_regressor.predict(new_pixels)
        
        # Predict presence/absence and add to features
        classifications = {}
        for col, classifier in self.classifiers.items():
            new_probas = classifier.predict_proba(new_pixels) 
            class_probas[col] = new_probas[:, 1]
            # classifications[fpcs_columns[col]] = pd.DataFrame(new_probas[:, 1], index = new_pixels.index)
        # classifications["water"] = pd.DataFrame(new_pixels[100], index = new_pixels.index)
            
        # Predict fractional abundances
        # predictions = {}
        for col, regressor in self.regressors.items():
            # test_predictions = regressor.predict(new_pixels)
            # class_regressions[col] = pd.DataFrame(test_predictions, index = new_pixels.index)
            class_regressions[col] = regressor.predict(new_pixels)
            
        return class_probas, new_pixels["water"],  class_regressions
        