import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale

def calc_stats(data : pd.DataFrame):
    """ Calculate the basic stats for each wavelength in the spectral data."""
    means = []
    std_devs = []
    min_vals = []
    max_vals = []
    counts = []
    data = data.select_dtypes(include="number")
    num_bands = len(data.columns)
    # Loop through each column (wavelength) in the dataset
    for i in range(num_bands):
        column_data = data.iloc[:, i]  # Selecting column i
        mean_val = np.mean(column_data)
        std_dev_val = np.std(column_data)
        min_val = np.min(column_data)
        max_val = np.max(column_data)
        count = len(column_data)

        # Append calculated statistics to respective lists
        means.append(mean_val)
        std_devs.append(std_dev_val)
        min_vals.append(min_val)
        max_vals.append(max_val)
        counts.append(count)

    # Create a dictionary of calculated statistics
    stats = {'mean': means, 'st_dev': std_devs, 'minimum': min_vals, 'maximum': max_vals, 'count' :counts}

    # Create a DataFrame from the statistics dictionary
    stats_df = pd.DataFrame(stats)

    # Reset index with proper name for the wavelength column
    stats_df = stats_df.reset_index(names='wavelength')
    stats_df['wavelength'] = stats_df['wavelength'] + 399

    # Drop the first column (index column from reset_index)
    stats_df = stats_df.iloc[:, 1:]

    # Calculate lower and upper bounds for plotting
    stats_df['lower'] = stats_df['mean'] - stats_df['st_dev']
    stats_df['upper'] = stats_df['mean'] + stats_df['st_dev']
    
    return stats_df


def scale_data (data, chosen_scaler) -> pd.DataFrame: 
    """Scale data using either min-max or standard scaling.
    
    Args: 
        data (pd.Dataframe): Dataframe of spectral data (no labels)
    
    Returns:
        pd.Dataframe: Scaled data.
        
    Raises:
        ValueError: If unrecognized scaler is chosen.
    
    """
    chosen_scaler = chosen_scaler.lower()
    if chosen_scaler == "minmax":
        scaled_data = pd.DataFrame(minmax_scale(data, axis = 1))
        return scaled_data
    elif chosen_scaler == "standard":
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data.T).T)
        return scaled_data
    else:
         raise ValueError("Invalid scaler. Chose either 'minmax' or 'standard.")


def crop_noisy_wavelengths(data, start_nm, end_nm):
    start_nm = str(start_nm)
    end_nm = str(end_nm)
    cropped = data.loc[:, start_nm:end_nm ]
    return cropped


def assign_targets_list(labels, group = "all"):
    if group == 'kelp':
        target_classes = ['ecklonia', 'cystophora', 'macrocystis', 'carpophyllum','gravel', 'grass', 'rock', 'durvillaea', 'rhodophyte', 'undaria', 'phyllospora','sand', 'acrocarpia', 'hormosira', 'mussels']
    elif group == "all":
        target_classes = labels["Class"].unique()
    elif group == "browns":
        target_classes = ['ecklonia', 'cystophora', 'macrocystis', 'carpophyllum','gravel', 'grass', 'rock', 'durvillaea', 'rhodophyte', 'undaria', 'phyllospora','sand', 'acrocarpia', 'hormosira', 'mussels']
    elif group == "farm": 
        target_classes = ['macrocystis', 'carpophyllum','rhodophyte', 'undaria']
    return target_classes 


def format_for_plotting(data, labels, bad_bands_list):
    """ Add labels to data and mask bad bands."""
    all_data = data.dropna(axis = 1, how = "any")
    labeled_data = pd.concat([labels, all_data.iloc[:, 1:]], axis = 1)
    mask = labeled_data.columns.isin(bad_bands_list)
    labeled_data.loc[:, mask] = np.nan
    return labeled_data


def filter_for_plotting(data,
                        labels,
                        scheme = "all",
                        target_sites = None,
                        target_setup = None,
                        ):
    #Define the filter parameters to chose target spectra
    
    target_classes = assign_targets_list(labels, scheme)

    #apply the filters
    all_targets = data[data["Class"].isin(target_classes)]
    if target_setup:
        all_targets = all_targets[all_targets["setup"].isin(target_setup)]
    if target_sites:
        all_targets = all_targets[all_targets["site"].isin(target_sites)]
    all_targets = all_targets.reset_index().drop("index", axis = 1)
    labels_columns = all_targets.select_dtypes(include = "object")
    targets_spectra = all_targets.select_dtypes(include = "number")
    return labels_columns, targets_spectra


def class_stats(target_spectra,
                target_labels,
                grouping = "Class",
                plot_by_site = True
                ):
    """ Calculate statistics for all target classes"""

    #Normalize spectra to a range of 0 to 1
    std_df = scale_data(target_spectra, "standard")
    
    #Choose to use raw or normalized data
    spectra_to_use = pd.concat([target_labels, std_df], axis = 1)

    #Start a dictionary to contain the results
    stats_dict = {}

    #Set number of wavelength columns
    #spectral_set_length = std_df[1]

    # Define the grouping from the chosen data
    basic_group = spectra_to_use[grouping].unique()

    # For each group, setup, site: make stats dict
    for species in basic_group:
        spec_targets = spectra_to_use[spectra_to_use["Class"] == species]
        if plot_by_site: 
            sites = spec_targets["site"].unique()
            for site in sites:
                spec_site = spec_targets[spec_targets["site"] == site]
                spec_site_reduced = spec_site.select_dtypes(include= "number")
                name = f"{species}_{site}"
                # Calculate the stats
                stats_dict[name]= calc_stats(spec_site_reduced)
        else:
            spec_reduced = spec_targets.select_dtypes(include= "number")
            name = f"{species}"
            # Calculate the stats
            stats_dict[name]= calc_stats(spec_reduced)
    return stats_dict


def prepping_chain(data, labels):
    """Data prep for  the plotting with default values."""
    scheme = "all" # "kelp", "browns", "farm"
    target_sites = ["new_zealand", "melbourne", "tasmania"]
    target_setup = ["handheld"]
    bad_bands_list = map(str,list(range(753, 769)))
    
        # Format the data
    formatted_data = format_for_plotting(data, labels, bad_bands_list)
    # Filter to targets of interest
    filtered_labels, filtered_data = filter_for_plotting(formatted_data,
                                                         labels,
                                                         scheme = scheme,
                                                         target_setup = target_setup,
                                                         target_sites = target_sites)
    # crop out noisy wavelengths
    cropped_data = crop_noisy_wavelengths(filtered_data, "400", "900")
    # Scale the data
    scaled_data = scale_data(cropped_data, "standard")
    # Calculate the class specific stats
    class_stats_dict = class_stats(scaled_data, filtered_labels, grouping = "Class", plot_by_site= True)
    return class_stats_dict


def plot_search_terms(search_words,
                      search_sites,
                      mode,
                      stats_dict,
                      start_wavelength=400,
                      save_plot = False):
    """ Plots the mean with standard deviation shaded of the search terms. Mode of either any or all to define search logic"""
    #make list of all combos of species, site, setup
    options = stats_dict.keys()

    # Filtered list of options for subset using list comprehension
    filtered_list = [item for item in options if mode(word in item for word in search_words)]
    if len(search_sites) != 0:
        filtered_list = [item for item in filtered_list if mode(word in item for word in search_sites)]

    sns.set_palette("tab20")
    plt.clf()
    for w in filtered_list:
        # Plot with error bars
        plotting_data = pd.DataFrame(stats_dict[w])
        plotting_data.index = plotting_data.index+start_wavelength        
        plt.plot(plotting_data.index, plotting_data['mean'], "-", label = w)  
        plt.fill_between(plotting_data.index, plotting_data['lower'], plotting_data['upper'], alpha=0.1)
    plt.tight_layout(pad = 4, w_pad= 1, h_pad= 1)
    plt.title("Mean reflectance spectra +/- st.dev")
    plt.xlabel('Wavelength')
    plt.legend(loc = "upper left")
    plt.ylabel('Log (Min-Max transformed reflectance)')
    plt.xticks([num for num in range(350, 950, 100)])
    if save_plot:
        plot_name = rf'C:\Users\s4770224\Documents\coding\Spectral_analysis\Plots\drafts\{search_words}.png'
        plt.savefig(plot_name)
    plt.show()
    return
