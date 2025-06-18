#spectral angle mapper from scratch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, precision_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def ingest_and_clean (data_path, labels_path):
    #prepare data for ingestion: Run once at start
    data_import = pd.read_csv(data_path)                    #INPUT REQUIRED
    labels_import = pd.read_csv(labels_path)   #INPUT REQUIRED

    labelled_data = pd.concat([labels_import,data_import.iloc[:,1:]], axis = 1)
    labelled_data = labelled_data.dropna(axis = 1, how = "any") #restrict to whats measured in all sets
    bad_classes = ["wr", "bryozoans", "background", "stray_light", "bad"]
    labelled_data = labelled_data[~labelled_data["Class"].isin(bad_classes)]
    return labelled_data
        
def calculate_master_averages(spectral_data, classes):
    averages_master={}
    for item in classes:
        subset = spectral_data[spectral_data["Class"] == item].drop("Class", axis =1)
        averages_master[item] = np.mean(subset, axis = 0)
    return averages_master

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

def plot_cm (true_classes, pred_classes, save_name, normalize = "true"):
    cm = confusion_matrix(true_classes, pred_classes, normalize = normalize)
    cm_masked = np.ma.masked_equal(cm, 0)
    rel_path = "Outputs\\SAM_results\\March2025\\"
    save_path = rel_path +save_name

    cm_df= pd.DataFrame(cm)
    cm_masked = cm_df.map(lambda v: str(round(v, 2)) if v >0 else "")

    # Optional: Visualize the confusion matrix using seaborn
    plt.close()
    sns.heatmap(cm, annot=cm_masked, fmt = "s", cmap='Blues', 
                xticklabels= unique_labels, 
                yticklabels= unique_labels,
                annot_kws={"size" : 6}, 
                vmin = 0, 
                vmax = 1
                )
    plt.tight_layout(pad = 4, w_pad= 1, h_pad= 1)
    plt.xlabel('Predicted Labels')
    plt.xticks(rotation = 30)
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {save_name}')
    plt.savefig(save_path)
    plt.show()

def assign_targets_list(group = "macros"):
    if group == 'kelp':
        target_classes = ['ecklonia', 'macrocystis', 'undaria', 'petalonia']
    elif group == "all":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'sand', 'rock', 'acrocarpia', 'hormosira', 'mussels','macrocystis', 'carpophyllum', 'gravel', 'grass', 'filamentous_rhodophyte', 'undaria', 'frondose_rhodophyte', 'shell_litter', 'ulva', 'sargassum', 'barnacle_shells', 'worm_castings', 'scytosiphon','petalonia']
    elif group == "macros":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'acrocarpia', 'hormosira', 'macrocystis', 'carpophyllum', 'filamentous_rhodophyte', 'undaria', 'frondose_rhodophyte', 'ulva', 'sargassum', 'scytosiphon','petalonia']
    elif group == "browns":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'acrocarpia', 'hormosira', 'macrocystis', 'carpophyllum', 'undaria', 'sargassum', 'scytosiphon','petalonia']
    elif group == "farm": 
        target_classes = ['macrocystis', 'carpophyllum','filamentous_rhodophyte', 'undaria']
    return target_classes

def format_data (data_df: pd.DataFrame, keep_classes: [], min_band = "400", max_band = "925", fit_min = "400", fit_max = "925", text_col_count = 3):
    """Subset and standardize the imported spectra for classification.
    Parameters:
    data_df: pd.Dataframe of spectra, with samples in rows and wavelengths in columns. All labels must be grouped in the first columns.
    keep_classes: List of target types being kept in the column labeled "Class"
    min_band : String -smallest wavelength to be kept
    max_band: String - Largest wavelength to be kept
    fit_min : String -smallest wavelength to be standardized to (i.e. low noise region)
    fit_max: String - Largest wavelength to be standardized to (i.e. low noise region)
    text_col_count: Int - Number of text(label) columns in the data
    
    Output:
    df_scaled: pd.Dataframe of scaled spectra, subset to keep list, with label columns leading
    """
    data = data_df[data_df["Class"].isin(keep_classes)]
    text_cols = data.iloc[:, :text_col_count]
    text_cols.shape
    df=data.loc[:, min_band:max_band]
    df.shape
    df = df.transpose()
    scaler = StandardScaler().fit(df.loc[fit_min: fit_max, :])
    df_scaled = pd.DataFrame(scaler.transform(df))
    df_scaled.transpose().set_index(df.columns, inplace= True)
    df_scaled.columns = data.index
    df_scaled.index = data.loc[:, min_band:max_band].columns
    df_scaled = pd.concat([text_cols, df_scaled.transpose()], axis = 1)
    return df_scaled


def sort_classes(df):
    # Make sort_order of classes
    sort_order = ['macrocystis','undaria','ecklonia','cystophora','carpophyllum','phyllospora', 'durvillaea','sargassum', 'petalonia', 'scytosiphon', 'acrocarpia','ulva','hormosira','filamentous_rhodophyte', 'frondose_rhodophyte','grass','mussels','worm_castings', 'barnacle_shells', 'shell_litter','sand', 'gravel', 'rock',]

    # Make a sort column with categorical type, sort to match the sort_order
    df["sort_column"] = df["Class"]
    df['sort_column'] = pd.Categorical(df['sort_column'], categories=sort_order, ordered=True)
    df = df.sort_values('sort_column')
    return df

data_file_path = r"Combined_analysis\MEL_NZ_TAS_spectra.csv"
labels_file_path = r"Combined_analysis\New_reflectance_labels.csv"

labelled_data = ingest_and_clean(data_file_path, labels_file_path)



#Filter for just the classes, setup and sites of interest
keep_classes = assign_targets_list("macros") #could be all, macros, kelp, browns, or farm    
keep_setup = ["handheld"]
keep_site = ["melbourne", "new_zealand", "tasmania"]

subset = format_data(pd.read_csv(data_file_path), keep_classes, min_band = "460", fit_min= "460")
subset = subset[subset["setup"].isin(keep_setup)]
subset = subset[subset["site"].isin(keep_site)]

sorted_df = sort_classes(subset)

# Make data and labels dataframes
data = sorted_df.select_dtypes(include = "number")
labels_new = sorted_df.select_dtypes(include = "object")
resampling = "none_all"
export_results = True
for c in range(0,1):
    classification_column = c
    #Run everytime you change groupings
    labels = labels_new.iloc[:,classification_column]  #choose what labelling is being used
    group_name = labels.name
    labels.name = "Class"
    data = data.select_dtypes(include = "number") #get rid of all text columns
    data = data.dropna(axis = 1, how = "any") #drop all wavelengths that are not measured in all sets
    data = pd.concat([labels, data], axis = 1)  #add back the labels being used 
    #data.head()     #view the data
    unique_labels = labels.unique()

    #Make figure output save_name
    fig_save_name = group_name + "_" + resampling + "_SAM"

    #Make numerical labels
    label_number_dict = {}  #intitiate a dictionary for the equivalencies
    num_labels = np.zeros(len(labels))  #initiate an ndarray for the numerical labels
    for i, label in enumerate(unique_labels, start=0):  # Assign numbers to each unique label
        label_number_dict[label] = i
    #label_number_dict           #view the dictionary

    for n in range(0, len(labels)):        #Assign each sample with a numerical label
        #assign = labels.iloc[n]  # Ensure labels has enough items
        num_labels[n] = label_number_dict.get(labels.iloc[n], -1)

    #Make master list of endmembers from averages
    averages_master = calculate_master_averages(data, unique_labels)

    #Make placeholders for the results
    results = np.zeros([len(data), len(unique_labels)], float)

    #Calculate the spectral angle for each sample against each endmember, with sample removal from class endmember
    for row in range (len(data)):
        averages_temp = averages_master         #make copy of endmembers
        sample_class = data.iloc[row, 0]        #identify class of sample being analysed
        sample_data = data.iloc[row, 1:]        #copy out the sample being analysed
        temp_data = np.delete(data, row, axis = 0)    #drop the sample
        class_subset = data[data["Class"] == sample_class].drop("Class", axis =1)   #subset to the class of the sample
        averages_temp[sample_class] = np.mean(class_subset, axis = 0)       #overwrite the temporary endmember for the sample class
        column_counter = 0
        for label in averages_temp:
            vector1 = sample_data
            vector2 = averages_temp[label]
            angle = calculate_arccosine_of_dot_product(vector1, vector2)
            results[row, column_counter] = angle
            column_counter += 1

    #identify which endmember was closest
    minimum_distance_class = np.argmin(results, axis = 1)

    #identify the distance to the closest endmember (looking for outliers)
    minimum_angle = np.min(results, axis = 1)

    accuracy = accuracy_score(num_labels, minimum_distance_class)
    balanced_accuracy = balanced_accuracy_score(num_labels, minimum_distance_class)
    recall = recall_score(num_labels, minimum_distance_class, average="weighted")
    precision = precision_score(num_labels, minimum_distance_class, average= "macro")
    print(f" The accuracy of {fig_save_name} is: {accuracy}")
    print(f" The balanced accuracy of {fig_save_name} is: {balanced_accuracy}")
    print(f" The recall of {fig_save_name} is: {recall}")
    print(f" The precision of {fig_save_name} is: {precision}")

    plot_cm(num_labels, minimum_distance_class, f"{fig_save_name}.svg")
    
    if export_results == True:
        results_df = pd.DataFrame(results)
        num_labels_df = pd.DataFrame(num_labels)
        minimum_distance_class_df = pd.DataFrame(minimum_distance_class)
        overall_results = pd.concat([num_labels_df, minimum_distance_class_df, results_df], axis = 1)
        overall_results.to_csv(rf".\Outputs\SAM_results\{fig_save_name}.csv")
        labels_dict_df= pd.DataFrame.from_dict(label_number_dict, orient = 'index')
        labels_dict_df.to_csv(rf".\Outputs\SAM_results\dictionary_{fig_save_name}.csv")
        averages_master_df=pd.DataFrame.from_dict(averages_master)
        averages_master_df.to_csv(rf".\Outputs\SAM_results\averages_{fig_save_name}.csv")