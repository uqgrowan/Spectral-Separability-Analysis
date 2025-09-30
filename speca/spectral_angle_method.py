import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
import auxilliary.auxiliary as aux

def calculate_arccosine_of_dot_product(vector1, vector2):
    """
    Calculate the arccosine of the dot product of two vectors,
    normalized by the product of their Euclidean norms.

    Parameters:
    vector1 (np.array): First input vector.
    vector2 (np.array): Second input vector.

    Returns:
    float: Arccosine of the normalized dot product, in radians.
    """
    # Compute the dot product
    dot_product = np.dot(vector1, vector2)

    # Compute the Euclidean norms (2-norms) of the vectors
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Compute the cosine of the angle between the vectors
    cosine_angle = dot_product / (norm1 * norm2)

    # Restrict the cosine value to the range [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Compute the arccosine of the cosine value
    angle = np.arccos(cosine_angle)

    return angle


class SAM:
    """
    Adapted spectral angle calculation to handle point-source data instead of images.
    """

    def __init__(self, spectra, labels, labels_col='Class', by_site = False):
        """ 
        Initialize the SAM class with data and labels to be classified.
        """
        self.spectra = spectra
        self.labels = labels
        self.labels_col = labels_col
        
        # Initialize empty variables for later use
        self.classes = list(self.labels[labels_col].unique())
        self.class_averages = None
        self.cosines_array = None
        self.numerical_labels = None
        self.prediction = None
        self.by_site = by_site
        

    def calc_spec_angles(self):
        """
        Calculate the angle between each sample and the class averages
        """
        # Make array of all class means
        self.class_averages = aux.calc_class_averages(self.spectra, self.labels, self.labels_col, average_by_site = self.by_site)

        # Choose labelling scheme
        if self.by_site is True:
            # labels = self.labels.loc[:,[self.labels_col, "site"]]
            # labels["class-site"] = labels.apply(lambda x: f"{x['Class']}-{x['site']}", axis=1)
            labels = self.labels.apply(lambda x: f"{x[self.labels_col]}-{x['site']}", axis=1)
            labels.name = self.labels_col
        else:
            labels = self.labels[self.labels_col]

        # Make dictionary of number label for each class
        self.classes= labels.unique()
        labels_num_dict = {v:k for k, v, in dict(enumerate(self.classes)).items()}

        # Combine labels and spectra
        data = pd.concat([labels, self.spectra], axis = 1)
        data = data.dropna(axis = 1, how = "any")

        # Make array of numerical labels
        num_labels = np.zeros(len(labels))
        
        # Assign num labels to each sample
        for n in range(len(labels)):
            num_labels[n] = labels_num_dict.get(labels.iloc[n], -1)
        self.numerical_labels = num_labels

        # Make placeholders for the results
        results = pd.DataFrame()

        # Calculate the sample-class spectral angle with sample removal
        for row in range(len(data)):
            # Isolate the sample and its class
            sample_index = data.index[row]
            averages_temp = self.class_averages
            sample_class = str(data.iloc[row, 0])
            sample_data = data.iloc[row, 1:]
            
            # Drop the sample
            temp_data = data.drop(index = sample_index, axis = 0) 

            # Subset to sample class
            class_subset = temp_data[temp_data[self.labels_col] == sample_class] 
            class_subset = class_subset.select_dtypes(include = "number")

            # Recalculate the class average
            averages_temp.loc[sample_class, :] = np.mean(class_subset, axis = 0)

            for label in self.classes:
                vector1 = sample_data
                vector2 = averages_temp.loc[label]
                angle = calculate_arccosine_of_dot_product(vector1, vector2)
                results.loc[row, label] = angle
                
        self.cosines_array = results

    def find_minimum_angle(self):
        """ Identify the minimum angle for each sample and return the class. """
        # Identify which endmember was closest
        minimum_distance_class = np.argmin(self.cosines_array, axis = 1)
        minimum_distance_class = pd.DataFrame(minimum_distance_class, columns = ["predicted_num"])
        self.prediction = minimum_distance_class

        #Make figure output save_name
        fig_save_name = self.labels_col + "_" + "SAM"

        accuracy = accuracy_score(self.numerical_labels, minimum_distance_class)
        balanced_accuracy = balanced_accuracy_score(self.numerical_labels, minimum_distance_class)
        recall = recall_score(self.numerical_labels, minimum_distance_class, average="weighted")
        precision = precision_score(self.numerical_labels, minimum_distance_class, average= "macro")
        f1 = f1_score(self.numerical_labels, minimum_distance_class, average = "macro")
        print(f" The accuracy of {fig_save_name} is: {accuracy}")
        print(f" The balanced accuracy of {fig_save_name} is: {balanced_accuracy}")
        print(f" The recall of {fig_save_name} is: {recall}")
        print(f" The precision of {fig_save_name} is: {precision}")
        print(f" The macro F1 score of {fig_save_name} is: {f1}")

    def plot_cm(self):
        """ Plot the confusion matrix of the classification"""

        # Make confusion matrix
        cm = confusion_matrix(self.numerical_labels, self.prediction["predicted_num"], normalize = "true")

        # Mask zero values
        #cm_masked = np.ma.masked_equal(cm, 0)
        cm_df= pd.DataFrame(cm)
        cm_masked = cm_df.map(lambda v: str(int(v*100)) if int(v*100) >0 else "")

        # Visualize the confusion matrix using seaborn
        plt.close()
        plt.figure(figsize=(12, 10))
        sns.set_theme(font = "Times New Roman")
        sns.heatmap(cm*100, annot=cm_masked, fmt = "s", cmap='Blues', 
                    xticklabels= self.classes, 
                    yticklabels= self.classes,
                    annot_kws={"size" : 12}, 
                    vmin = 0, 
                    vmax = 100
                    )
        plt.tight_layout(pad = 4, w_pad= 1, h_pad= 1)
        plt.xlabel('Predicted Labels')
        plt.xticks(rotation = 30)
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix for {self.labels_col} - SAM')
        plt.savefig("./plots/SAM/SAM_cm.svg", bbox_inches='tight', dpi=600)
        plt.show()

    def sam_chain(self):
        self.calc_spec_angles()
        self.find_minimum_angle()
        self.plot_cm()
