#spectral angle mapper from scratch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def name_results_columns (label_dictionary, results_toRename):
    classes_list = list(label_dictionary.iloc[:,0])
    columns_list = []
    columns_list.append("true_class")
    columns_list.append("pred_class")
    for f in range(0, len(classes_list)):
        columns_list.append(classes_list[f])
    results_toRename.columns = [columns_list]
    return results_toRename
def assign_true_pred_angles(results):
    angle_to_true = []
    angle_to_pred = []
    n_samples = results.shape[0]

    for r in range(0, n_samples):
        angle_to_true.append(results.iloc[r, int(results.iloc[r,0])+2])
        angle_to_pred.append(results.iloc[r, int(results.iloc[r,1])+2])

    results["angle_to_true"] = angle_to_true
    results["angle_to_pred"] = angle_to_pred
    return results   
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

    # Clamp the cosine value to the range [-1, 1] to avoid potential numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Compute the arccosine of the cosine value
    angle = np.arccos(cosine_angle)

    return angle

results_file_path = r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Outputs\SAM_results\Class_none_SAM.csv"
labels_dictionary_file_path = r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Outputs\SAM_results\dictionary_Class_none_SAM.csv"
averages_file_path = r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Outputs\SAM_results\averages_Class_none_SAM.csv"

#inport data
results_import = pd.read_csv(results_file_path, index_col=0)
label_num_dict = pd.read_csv(labels_dictionary_file_path)
averages_master = pd.read_csv(averages_file_path, index_col = 0)

#rename results columns to useful labels
results = name_results_columns(label_num_dict, results_import)
results = assign_true_pred_angles(results)

#Create random scatter to avoid all points overlapping
variability_1 = np.random.normal(0.0, 0.1, size=results.shape[0])
variability_2 = np.random.normal(0.0, 0.1, size=results.shape[0])
results["var_true_class"] = results["true_class"].squeeze().add(variability_1)
results["var_pred_class"] = results["pred_class"].squeeze().add(variability_2)

#create mask of correct and incorrect classifications
correct = results["true_class"].squeeze() == results["pred_class"].squeeze()
incorrect = ~correct

#SAM Scatter plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.clf()
plt.set_cmap("nipy_spectral") #gnuplot" , "nipy_spectral"
#plt.scatter(np.array(variability_1)[ko_correct], np.array(angle_to_true)[ko_correct], c=minimum_angle[ko_correct], alpha = 0.6, label = "correct")
#plt.scatter(np.array(variability_2)[ko_incorrect], np.array(angle_to_true)[ko_incorrect], c=minimum_angle[ko_incorrect], alpha = 1, label = "incorrect")

#plt.scatter(y = np.array(angle_to_true)[correct],x = minimum_angle[correct], c= "gray", marker = 'x', alpha= 0.2, label = "correctly classified")
#plt.scatter(y = np.array(angle_to_true)[incorrect],x =minimum_angle[incorrect], c= num_labels[incorrect], s = 120, marker = "o", label = "true class")
#plt.scatter(y = np.array(angle_to_true)[incorrect],x = minimum_angle[incorrect], c= minimum_distance_class[incorrect], s = 20, label = "predicted class")

plt.scatter(y = results["angle_to_true"][correct],x = results["angle_to_pred"][correct], c= "gray", marker = 'x', alpha= 0.2, label = "correctly classified")
plt.scatter(y = results["angle_to_true"][incorrect],x = results["angle_to_pred"][incorrect], c= results["true_class"][incorrect], s = 120, marker = "o", label = "true class")
plt.scatter(y = results["angle_to_true"][incorrect],x = results["angle_to_pred"][incorrect], c= results["pred_class"][incorrect], s = 20, label = "predicted class")
#plt.scatter(var_num_labels, angle_to_true, c=minimum_distance_class, alpha = 0.4)
#plt.scatter(var_num_labels[incorrect], np.array(angle_to_true)[incorrect], c=minimum_distance_class[incorrect], alpha = 1)
plt.xlabel('Minimum Spectral Angle')
plt.ylabel('Angle to true class')
plt.title('Spectral Angle mapping')
plt.yscale("log")
plt.legend()
plt.show()


#SAM Confusion Matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#######START HERE TOMORROW: you are currently working on getting this section to use the results df and the averages_master df as outputted by the SAM script
averages_results = {}
for col1 in averages_master.columns:
    angle_list = []
    vector1 = averages_master[col1]
    for col2 in averages_master.columns:
        vector2 = averages_master[col2]
        angle = calculate_arccosine_of_dot_product(vector1, vector2)
        angle_list = np.append(angle_list, angle)
    averages_results[col1] = angle_list

averages_results_df = pd.DataFrame(averages_results)
averages_results_df.head()
angle_diff = results["angle_to_true"] - results["angle_to_pred"]

fig, ax = plt.subplots()
im = ax.imshow(np.sqrt(averages_results_df), cmap = "binary")
cbar = fig.colorbar(im, ax = ax)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(averages_results_df)))
ax.set_yticks(np.arange(len(averages_results_df)))

# Rotate the tick labels and set their alignment.
plt.scatter(results["var_true_class"], results["var_pred_class"], c= results["angle_to_pred"], cmap = "plasma_r", alpha = 0.8,)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va = "top", rotation_mode="default")
plt.colorbar(label = "Spectral Angle between sample to predicted class")
for b in range(0, 14):
    b  = b+0.5
    plt.axhline(y =b, c = "w", lw = 0.5 )
for b in range(0, 14):
    b  = b+0.5
    plt.axvline(x =b, c= "w", lw = 0.5 )
ax.set_xlabel = ("True Class")
ax.set_ylabel = ("Predicted Class")
ax.set_title("Spectral Angle between class means")
fig.tight_layout()
plt.show()
