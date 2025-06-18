#Spectral dimension reduction testing

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from sklearn.datasets import load_iris
#iris = load_iris()
#iris_array = iris["data"]
#iris_df = pd.DataFrame(iris_array)
#iris_df.head()

spectra = pd.read_csv('C:\\Users\\s4770224\\Documents\\Work\\Spectral analysis\\Mine\\Melbourne_spectra_transposed.csv')
#print(spectra.head())

num_df = spectra.drop("Class", axis = 1)
#print(num_df.head())

#y = spectra["Class"]
#print(y.head())

#x= spectra.drop("Class", axis = 1)
from sklearn.manifold import TSNE
m = TSNE(learning_rate= 500)
tsne_features = m.fit_transform(num_df)

tsne_features[1:4,:]
tsne_df = pd.DataFrame(tsne_features)
num_df["x"] = tsne_df.loc[:, 0]
num_df["y"] = tsne_df.loc[:, 1]

#print(num_df.head())

sns.scatterplot(x = "x", y = "y", data = num_df)
plt.show()

#MAKE TESTING AND TRAINING SETS
dep = spectra["Class"]
indep= spectra.drop("Class", axis = 1)
#dep = iris["target"]     #troubleshooting with Iris dataset
#indep = iris_df          #troubleshooting with Iris dataset


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(indep, dep, test_size= 0.2)
print(f"{x_test.shape[0]} rows in test set vs. {x_train.shape[0]} in training set, {x_test.shape[1]} features.")

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.95)
sel.fit ( indep / indep.mean( ) )
mask = sel.get_support()
sum(mask)

selected = indep.loc[:, mask]
selected_corr = selected.corr().abs()
cmap = sns.diverging_palette (h_neg = 10, h_pos = 240, as_cmap = True)
map_mask = np.triu (np.ones_like(selected_corr, dtype = bool))
sns.heatmap(selected_corr, center = 0, cmap = "crest", vmin=0, vmax = 1, linewidths = 0.5, mask = map_mask)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler  #
scaler = StandardScaler()  #start and instance of Standard Scaler 
x_train_std = scaler.fit_transform(x_train)  #fit and scale the data 
lr = LogisticRegression ( max_iter = 1000) # start an instance of logistic Regression
lr.fit(x_train_std, y_train)  # train the model 
x_test_std = scaler.transform(x_test)  #transform the testing set the same as the training set
y_pred = lr.predict(x_test_std) #apply trained model to testing data
print(accuracy_score(y_test, y_pred))  #check accuracy



rfe = RFE(estimator = LogisticRegression(max_iter = 1000), n_features_to_select = 70, step = 20, verbose = 1) #dropuntil only n features
rfe.fit(x_train_std, y_train) #
indep.columns[rfe.support_]   #print array of features kept
print(dict(zip(indep.columns, rfe.ranking_)))  # print dictionary of the order features were dropped (lower = kept)

print(accuracy_score(y_test, rfe.predict(x_test_std))) # check accuracy with only n features

kept = indep.columns[rfe.support_]
most_info = indep.loc[:, kept]
corr = most_info.corr()
mask = np.triu (np.ones_like(corr, dtype = bool))  # masks out the upper triangle and the diagonals
cmap = sns.diverging_palette (h_neg = 10, h_pos = 240, as_cmap = True) #define properties for your heat map
sns.heatmap(most_info.corr(), center = 0, cmap = cmap, linewidths = 0.2, mask = mask, fmt = ".2f") #add mask
plt.show()



#removing highly correlated values
corr_df = most_info.corr().abs()  # use absolute values to also get strong negative correlations
mask = np.triu(np.ones_like(corr_df, dtype = bool))
tri_df = corr_df.mask(mask)  # mask the duplicate values and diagonals in the corr matrix
to_drop = [c for c in tri_df.columns if any (tri_df [ c ] > 0.95)] #makes a list of columns in the df with corr over 0.95
print(to_drop) # view the list of features to drop
cleaned_most_info = most_info.drop(to_drop, axis = 1)
corr = cleaned_most_info.corr()
mask = np.triu (np.ones_like(corr, dtype = bool))  # masks out the upper triangle and the diagonals
cmap = sns.diverging_palette (h_neg = 10, h_pos = 240, as_cmap = True) #define properties for your heat map
sns.heatmap(cleaned_most_info.corr(), center = 0, cmap = cmap, linewidths = 0.2, mask = mask, fmt = ".2f") #add mask
plt.show()
cleaned_most_info.head()

#####TESTING ON REDUCED DIMS SET #########

indep= cleaned_most_info

x_train, x_test, y_train, y_test = train_test_split(indep, dep, test_size= 0.2)
print(f"{x_test.shape[0]} rows in test set vs. {x_train.shape[0]} in training set, {x_test.shape[1]} features.")

scaler = StandardScaler()  #start and instance of Standard Scaler 
x_train_std = scaler.fit_transform(x_train)  #fit and scale the data 
lr = LogisticRegression ( max_iter = 1000) # start an instance of logistic Regression
lr.fit(x_train_std, y_train)  # train the model 
x_test_std = scaler.transform(x_test)  #transform the testing set the same as the training set
y_pred = lr.predict(x_test_std) #apply trained model to testing data
print(accuracy_score(y_test, y_pred))  #check accuracy

corr_cleaned = cleaned_most_info.corr()
corr_cleaned.to_csv("correlation_output.csv")

#########################FEATURE EXTRACTTION#####################

Scaler = StandardScaler()
std_df = pd.DataFrame(scaler.fit_transform(indep), columns = indep.columns)
 
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

pipe = Pipeline ( [ 
    ("scaler", StandardScaler ( )),  #scale your data to mean 0, var 1
	("reducer", PCA ( n_components = 0.95) ),    # extract first 12 PCA components. Using number between 0-1 will keep components until that variance is met  (i.e. 0.9 will keep enough comps to explain 90% variance)
	( "classifier" , RandomForestClassifier () ), # train a random forest classifier on your data
	] )

pipe.fit(x_train, y_train)
pipe["reducer"].explained_variance_ratio_
print(pipe.score(x_test, y_test))
components = pd.DataFrame(pipe["reducer"].components_)
components.to_csv("components_output.csv")
