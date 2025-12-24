import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class Unmix_Pixels:
    
    def __init__(self, x_train, y_train, x_test, y_test, 
                 presence_absence_thresh = 0.5):
        self.x_train = x_train.copy()
        self.y_train = y_train.copy().set_index(x_train.index)
        self.x_test = x_test.copy()
        self.y_test = y_test.copy().set_index(x_test.index)
        self.presence_thresh = presence_absence_thresh
        self.water_regressor = None
        self.classifiers = None
        self.regressors = None
    
    def classify_and_append(self, classifier_params, plot):
        
        presence_probabilities = {
            "test": {},
            "train": {}
        }
        
        # Predict water fraction and add to features
        water_rfr = RandomForestRegressor(**classifier_params).fit(self.x_train, self.y_train.iloc[:, -1])
        self.x_train[100] = water_rfr.predict(self.x_train)
        self.x_test[100] = water_rfr.predict(self.x_test)

        self.water_regressor = water_rfr
        
        if plot == "water" or plot == "both":    
            plt.scatter(x = self.y_test.iloc[:, -1], y = self.x_test[100], marker = 'o', alpha = 0.4, c= "teal")
            plt.title("Water fraction regression")
            plt.axline((0,0), slope = 1, color = "black", linestyle = "--")
            plt.show()
        
        else:
            print("Skipping water regressor plot...")
        
        
        # Predict presence/absence and add to features
        classifiers = {}
        for col in range(len(self.y_train.columns[:-1])):
            rfc = RandomForestClassifier().fit(self.x_train.loc[:, :100], self.y_train.iloc[:,col]>0)

            test_probas = rfc.predict_proba(self.x_test.loc[:, :100]) 
            train_probas = rfc.predict_proba(self.x_train.loc[:, :100]) 
            print(f"Classification test score: {rfc.score(self.x_test.loc[:, :100], self.y_test.iloc[:, col]>0)}")           
            self.x_train[101+col] = train_probas[:,1] > self.presence_thresh
            self.x_test[101+col] = test_probas[:, 1] > self.presence_thresh
            presence_probabilities["train"][self.y_train.columns[col]] = train_probas[:, 1]
            presence_probabilities["test"][self.y_train.columns[col]] = test_probas[:, 1]

            classifiers[self.y_train.columns[col]] = rfc
            
            if plot == "classifiers" or plot == "both":
                
                plt.scatter(x = self.y_test.iloc[:, col], 
                            y = test_probas[:, 1], marker = 'o', alpha = 0.4, 
                            c = test_probas[:, 1] > self.presence_thresh, 
                            label = "Probability of Presence")
                plt.scatter(x = self.y_test.iloc[:, col],
                            y = test_probas[:, 1]> 0.5, 
                            alpha = 0.4,
                            c="red",
                            label = "Hard classification")
                plt.legend()
                plt.xlabel("True Fractional Abundance")
                plt.ylabel("Predicted Presence Probability")
                plt.show()
            else:
                print(f"Not plotting {self.y_test.columns[col]} classifier...")
            
            
        self.classifiers = classifiers       
        presence_probabilities["train"]["water"] = self.x_train[100]
        presence_probabilities["test"]["water"] = self.x_test[100]

        return presence_probabilities
                

    def filter_and_regress(self, plot, regressor_params):
        
        class_predictions = {
            "train" : {},
            "test" : {}
        }
        regressors = {}
        for col in range(len(self.y_train.columns[:-1])):
            
            # Filter to only pixels with class present
            test_presence_mask = self.x_test.loc[:, 101+col] > 0.5
            train_presence_mask = self.x_train.loc[:, 101+col] > 0.5
            print(f"{self.y_train.columns[col]} present in {sum(test_presence_mask)} out of {len(test_presence_mask)} test pixels")

            print(f"Training set for {self.y_train.columns[col]} has {self.y_train.shape[0]} samples")
            
            #Train regressor on all pixels
            reg = RandomForestRegressor(**regressor_params).fit(self.x_train.loc[train_presence_mask,:100], self.y_train.loc[train_presence_mask, self.y_train.columns[col]])
            
            test_predictions = pd.Series(0.0, index = self.x_test.index)
            present_test_idx  = self.x_test.loc[test_presence_mask].index
            test_predictions[present_test_idx] = reg.predict(self.x_test.loc[test_presence_mask, :100])
            train_predictions = pd.Series(0.0, index = self.x_train.index)
            present_train_idx  = self.x_train.loc[train_presence_mask].index
            train_predictions[present_train_idx] = reg.predict(self.x_train.loc[train_presence_mask, :100])
            
            class_predictions["train"][self.y_train.columns[col]] = pd.DataFrame(train_predictions, index = self.x_train.loc[train_presence_mask].index)
            class_predictions["test"][self.y_train.columns[col]] = pd.DataFrame(test_predictions, index = self.x_test.loc[test_presence_mask].index)
            
            regressors[self.y_train.columns[col]] = reg
            
            #Predict on only present pixels
            present_predictions_test = reg.predict(self.x_test.loc[test_presence_mask, :100])
            
            if plot == "regressors" or plot == "both":
                
                plt.scatter(x = self.y_test.loc[test_presence_mask, self.y_train.columns[col]], 
                            y = present_predictions_test, marker = 'o', alpha = 0.4, c= self.x_test.loc[test_presence_mask, 100], label = self.y_test.columns[col])
                plt.legend()
                plt.xlabel("True Fractional Abundance")
                plt.ylabel("Regression Predicted FPC - ONLY CLASS PRESENT PIXELS")
                plt.show()
                
                plt.scatter(x = self.y_test.iloc[:, col], 
                            y = test_predictions, marker = 'x', alpha = 0.4, c= self.x_test.loc[:, 100], label = self.y_test.columns[col])
                plt.legend()
                plt.xlabel("True Fractional Abundance")
                plt.ylabel("Regression Predicted FPC - ALL PIXELS")
                plt.show()
            else:
                print(f"Not plotting {self.y_test.columns[col]} regression...")
        self.regressors = regressors
        return class_predictions

    def unmix_new_data(self, new_pixels, new_fpcs):
        new_pixels = new_pixels.copy()
        new_fpcs = new_fpcs.copy().set_index(new_pixels.index)
        
        # Predict water fraction and add to features
        new_pixels[100] = self.water_regressor.predict(new_pixels)
        
        # Predict presence/absence and add to features
        classifications = {}
        for col, classifier in enumerate(self.classifiers.values()):
            new_probas = classifier.predict_proba(new_pixels.loc[:, :100]) 
            new_pixels[101+col] = new_probas[:, 1] > self.presence_thresh
            classifications[new_fpcs.columns[col]] = pd.DataFrame(new_probas[:, 1], index = new_pixels.index)
        classifications["water"] = pd.DataFrame(new_pixels[100], index = new_pixels.index)
            
        # Predict fractional abundances
        predictions = {}
        for col, regressor in enumerate(self.regressors.values()):
            test_predictions = regressor.predict(new_pixels.loc[:, :100])
            predictions[new_fpcs.columns[col]] = pd.DataFrame(test_predictions, index = new_pixels.index)
            
        return classifications, predictions
        