import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale

def calc_stats(data : pd.DataFrame):
    """
    Calculate statistical measures for spectral data.
    
    Args:
        data: DataFrame containing spectral measurements
        
    Returns:
        DataFrame containing calculated statistics
    """
    numeric_data = data.select_dtypes(include="number")
        
    stats = pd.DataFrame({
        'mean': numeric_data.mean(),
        'st_dev': numeric_data.std(),
        'minimum': numeric_data.min(),
        'maximum': numeric_data.max(),
        'count': numeric_data.count(),
        'lower': numeric_data.mean() - numeric_data.std(),
        'upper': numeric_data.mean() + numeric_data.std()
    })
        
    return stats

def subset_to_targets(spectral_data, label_data, group = "all"):
    if group == 'kelp':
        target_classes = ['ecklonia', 'macrocystis', 'undaria']
        target_mask = label_data["Class"].isin(target_classes)
    elif group == "all":
        target_classes = label_data["Class"].unique()
        target_mask = label_data["Class"].isin(target_classes)
    elif group == "browns":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'acrocarpia', 'hormosira', 'macrocystis', 'carpophyllum', 'undaria', 'sargassum', 'scytosiphon', 'petalonia']
        target_mask = label_data["Class"].isin(target_classes)
    elif group == "farm": 
        target_classes = ['macrocystis', 'carpophyllum','filamentous_rhodophyte', 'undaria']
        target_mask = label_data["Class"].isin(target_classes)
    spectra = spectral_data[target_mask]
    labels = label_data[target_mask]
    return spectra, labels

def set_bands (sensor, spectral_data):
    if sensor == "asd":
        bands = spectral_data.columns.astype("int")
        bad_bands = [num-325  for num in range(753, 769)]
    else:
        bands = [i for i in range(0, spectral_data.shape[1])]
        bad_bands = []
    return bands, bad_bands

def scale_data (data, chosen_scaler):
    if chosen_scaler == "minmax":
        data = pd.DataFrame(minmax_scale(spectra, axis = 1))
    elif chosen_scaler == "standard":
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(spectra.T).T)
    else:
        print("Invalid scaler. Chose either 'minmax' or 'standard.'")
    return data

def prep_spectra_for_plotting(grouping, targets_subset = "all", scaler = "minmax", sensor = "asd", spectral_data = spectra, label_data = labels):
    '''
    Overall function to prep spectral data for plotting using the functions: subset_to_targets, set_bands, scale_data, and calc_stats.
    Adaptable for various sensors
    
    Args:
        grouping: (str) Class, order, colour, ko, kom, or bom.
        targets_subset: (str) all, farm, kelp, or browns
        scaler: (str) minmax or standard
        sensor: (str) Default "asd". any others are non-specific
        spectral_data: (DataFrame) Reflectances (samples x features)
        label_data: (DataFrame) Labels (samples x groupings)
        
    Outputs: bands list, bad bands list, and dictionary of stats for each unique label in the grouping
    '''
    
    #filter for only classes of interest, i.e. targets
    targets_spectra, targets_labels = subset_to_targets(spectral_data, label_data, group = targets_subset)

    #make list of band numbers for plotting later
    bands, bad_bands = set_bands(sensor, targets_spectra)


    #scale data and concat with labels
    spectra_scaled = scale_data(targets_spectra, scaler)
    data = pd.concat([targets_labels, spectra_scaled], axis = 1)

    #Choose grouping column of interest
    grouping_column = grouping
    unique_labels = targets_labels[grouping_column].unique()

    #Calculate stats for all labels
    stats_dict = {}
    for label in unique_labels:
        subset = data[data[grouping_column] == label]
        label_stats = calc_stats(subset)
        stats_dict[label] = label_stats
    return bands, bad_bands, stats_dict

def plot_all_labels(grouping, bands_list = bands, bad_bands_list = bad_bands, stats_dictionary = stats_dict):
    '''
    Plot the mean +- standard deviation for each unique label in the grouping.
    
    Args:
        grouping: (str) Class, order, colour, ko, kom, or bom.
        bad_bands_list : (list)
        bands : (list)
        
    Outputs: 
        Displays plot that can be saved in the window.
    '''
    #plot all labels
    for key in stats_dictionary.keys():
        mask = stats_dictionary[key].index.isin(bad_bands_list)
        stats_dictionary[key].loc[mask, :] = np.nan
        plt.plot(bands, stats_dictionary[key]['mean'], "-", 
                        label=f"{key}")
        plt.fill_between(x = bands_list, 
                        y1 = stats_dictionary[key]['lower'].values,
                        y2 = stats_dictionary[key]['upper'].values, 
                        alpha=0.1)
    plt.legend(loc="upper left")
    plt.tight_layout(pad = 4, w_pad= 1, h_pad= 1)
    plt.title(f"{grouping.capitalize()}")
    plt.xlabel('Band')
    plt.ylabel('Min-Max transformed reflectance')
    plt.ylim(-0.2, 1.2)
    plt.xticks()
    plt.show()

#import the data being used if not already available
spectra = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\Resampled\superdove.csv").select_dtypes("float")
labels = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\New_reflectance_labels.csv")

bands, bad_bands, stats_dict = prep_spectra_for_plotting("colour", targets_subset = "all", sensor = "lan")
plot_all_labels('colour')