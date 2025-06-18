#Spectral classification pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def feature_extraction_pipeline(path:str, class_labels, test_split: float,  n_comps: float, dep_var_name= "Class", preprocessing = "standard scaler", reducer = "pca", classifier= "random forest classifier"):
	"""Run a feature extraction pipeline on a specified spectral dataset. Pipeline includes standardization, feature extraction and classification. Outputs the accuracy of the pipeline on a test dataset. Functions available to the pipeline are defined below int he dictionary 'functions' to allow them to be callable. *** Note that other functions need to be imported before use.*** 
	Preprocessing, feature extraction and classifier are defaults set to Standard Scaler, PCA, and RFC"""
	spectra = pd.read_csv(path)

	#MAKE TESTING AND TRAINING SETS
	#dep = spectra[dep_var_name]  #make dependent variable set
	dep = class_labels  #use a set of labels not in spectra data file, i.e. for alternate classification schemes
	indep= spectra.drop(dep_var_name, axis = 1) #make independent variables data

	x_train, x_test, y_train, y_test = train_test_split(indep, dep, test_size= test_split)  # 80% training, 20% testing
	#print(f"{x_test.shape[0]} rows in test set vs. {x_train.shape[0]} in training set, {x_test.shape[1]} features.")

	functions = {
		"logistic regression" : LogisticRegression(),
		"random forest classifier" : RandomForestClassifier(),
		"pca" : PCA(n_components = n_comps),
		"kpca" : KernelPCA(n_components = n_comps),
		"standard scaler" : StandardScaler(),
	}
	pipe = Pipeline ( [ 
		("scaler", functions[preprocessing]),  #scale your data to mean 0, var 1
		("reducer", functions[reducer] ),    # extract first n PCA components. 0<n<1 will keep components until that variance is met (i.e. 0.9 = 90%)
		( "classifier", functions[classifier] ), # train a random forest classifier on your data
		] )
	pipe.fit(x_train, y_train)
	#print(f"the testing accuracy is {pipe.score(x_test, y_test)}. The training accuracy is {pipe.score(x_train, y_train)}")
	#print(f"The classification accuracy on the test set is {pipe.score(x_test, y_test)}, the accuracy on the training set is {pipe.score(x_train, y_train)}.")
	#print(pipe["reducer"].explained_variance_ratio_.cumsum())
	#print(pipe["reducer"].components_)
	return pipe.score(x_test, y_test)

path = 'C:\\Users\\s4770224\\Documents\\Work\\Spectral analysis\\Mine\\Melbourne_spectra_transposed_renamed.csv' # samples in rows,
labels = pd.read_csv("C:\\Users\\s4770224\\Documents\\coding\\Spectral analysis\\Melbourne_spectra_class_labels.csv")
labels.head()
class_labels = labels.iloc[:, 2]
#dep_var_name = "Class"
test_split = 0.2
#preprocessing = "standard scaler"
#classifier = "random forest classifier"
#reducer = "pca"
n_comps = 0.94

### Get single run accuracy
accuracy = feature_extraction_pipeline(path, class_labels, test_split, n_comps)
print(accuracy)

### Get average accuracy over many runs
accuracies = []
for i in range (0, 50):  #100 runs is about 15 secsond to run for PCA, 12 for kPCA
	run = feature_extraction_pipeline(path, class_labels, test_split, n_comps)
	accuracies.append(run)
sum(accuracies)/len(accuracies)



spectra = pd.read_csv(path)

#MAKE TESTING AND TRAINING SETS
#dep = spectra[dep_var_name]  #make dependent variable set
dep = class_labels  #use a set of labels not in spectra data file, i.e. for alternate classification schemes
indep= spectra.drop(["Class"], axis = 1) #make independent variables data


#run again and make a confusion matrix
x_train, x_test, y_train, y_test = train_test_split(indep, dep, test_size= 0.4)
rfe = RandomForestClassifier()
rfe.fit(x_train, y_train)
predictions = rfe.predict(x_test)
cm = confusion_matrix(y_test, predictions, labels= rfe.classes_)
bm = confusion_matrix(y_test, predictions, labels= rfe.classes_)
am = (cm+bm)/2

disp = ConfusionMatrixDisplay(confusion_matrix=am, display_labels=rfe.classes_)

disp.plot()
plt.show()

am = (cm+bm)/2



###Get accuracy over many runs for many different n_comps values to see what is optimal, output as dictionary then plotted
iterations = range(900, 1000, 2)
accuracy_assessment = {}
for s in iterations:
	n = s/1000
	accuracies = []
	for i in range (0, 100):  #100 runs is about 15 secsond to run for PCA, 12 for kPCA
		run = feature_extraction_pipeline(path, class_labels, test_split, n)
		accuracies.append(run)
	accuracy_assessment[n] = accuracies
		
assess_df = pd.DataFrame.from_dict(accuracy_assessment, orient = 'index')
components = accuracy_assessment.keys()
assess_df["components"] = components
melted = assess_df.melt(id_vars = "components")
assess_df.head()

sns.scatterplot(data = melted, x = "components", y = "value", alpha = 0.5)
plt.show()

melted.to_csv("pca_accuracies_100_runs.csv")



#Extract PCA components
def feature_extraction_components(path:str, class_labels, dep_var_name :str, test_split: float, preprocessing :str, reducer:str, n_comps: float, classifier:str):
	"""Run a feature extraction pipeline on a specified spectral dataset. Pipeline includes standardization, feature extraction and classification. Outputs the accuracy of the pipeline on a test dataset. Functions available to the pipeline are defined below int he dictionary 'functions' to allow them to be callable. *** Note that other functions need to be imported before use.*** """
	spectra = pd.read_csv(path)

	#MAKE TESTING AND TRAINING SETS
	#dep = spectra[dep_var_name]  #make dependent variable set
	dep = class_labels  #use a set of labels not in spectra data file, i.e. for alternate classification schemes
	indep= spectra.drop(dep_var_name, axis = 1) #make independent variables data

	x_train, x_test, y_train, y_test = train_test_split(indep, dep, test_size= test_split)  # 80% training, 20% testing
	#print(f"{x_test.shape[0]} rows in test set vs. {x_train.shape[0]} in training set, {x_test.shape[1]} features.")

	functions = {
		"logistic regression" : LogisticRegression(),
		"random forest classifier" : RandomForestClassifier(),
		"pca" : PCA(n_components = n_comps),
		"kpca" : KernelPCA(n_components = n_comps),
		"standard scaler" : StandardScaler(),
	}
	pipe = Pipeline ( [ 
		("scaler", functions[preprocessing]),  #scale your data to mean 0, var 1
		("reducer", functions[reducer] ),    # extract first n PCA components. 0<n<1 will keep components until that variance is met (i.e. 0.9 = 90%)
		( "classifier", functions[classifier] ), # train a random forest classifier on your data
		] )
	pipe.fit(x_train, y_train)
	#print(f"the testing accuracy is {pipe.score(x_test, y_test)}. The training accuracy is {pipe.score(x_train, y_train)}")
	#print(f"The classification accuracy on the test set is {pipe.score(x_test, y_test)}, the accuracy on the training set is {pipe.score(x_train, y_train)}.")
	#print(pipe["reducer"].explained_variance_ratio_.cumsum())
	return pipe["reducer"].components_

components_matrix = feature_extraction_components(path, class_labels, test_split, n_comps)
components_df = pd.DataFrame(components_matrix)
components_df = components_df.transpose()
components_df_renamed = components_df.reset_index(names = "wavelength")
components_df_renamed["wavelength"] = components_df_renamed["wavelength"] + 400
components_df_renamed.head()

#components_df_renamed.rename(columns={"wavelength":"wavelength", "0":"PCA1", "1":"PCA2", "2":"PCA3", "3": "PCA4", "4":"PCA5", "5":"PCA6", "6":"PCA7", "7":"PCA8", "8":"PCA9", "9":"PCA10", "10":"PCA11", "11":"PCA12", "12":"PCA13", "13":"PCA14", "14":"PCA15", "15":"PCA16", "16":"PCA17", "17":"PCA18","18":"PCA19","19":"PCA20"}) #rename columns to pca components if desired

#Plotting featuresall on one plot
sns.lineplot(data = components_df_renamed, x = "wavelength", y= 0, label = " PCA 1", markers= False)
sns.lineplot(data = components_df_renamed, x = "wavelength", y= 1, label = " PCA 2", markers= False)
sns.lineplot(data = components_df_renamed, x = "wavelength", y= 2, label = " PCA 3", markers= False)
sns.lineplot(data = components_df_renamed, x = "wavelength", y= 3, label = " PCA 4", markers= False)
sns.lineplot(data = components_df_renamed, x = "wavelength", y= 4, label = " PCA 5", markers= False)
sns.lineplot(data = components_df_renamed, x = "wavelength", y= 5, label = " PCA 6", markers= False)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Contribution to Component")
plt.legend()
plt.show()

#make them on a facet grid
comps_melted = components_df_renamed.melt(id_vars = "wavelength", var_name = "component", value_name = "contribution")
comps_melted.head()
comps_melted_8 = comps_melted[comps_melted["component"] <8]
sns.relplot(data=comps_melted_8, x="wavelength", y="contribution", col="component", col_wrap = 4, kind="line")
plt.tight_layout()
plt.show()

