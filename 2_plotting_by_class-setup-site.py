import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

def calc_stats(data : pd.DataFrame):
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


def standardize_data(df : pd.DataFrame, sensor):
    big=pd.DataFrame()
    lil = pd.DataFrame()
    df_std = pd.DataFrame()
    if sensor == "sphere":
        big = np.max(df.iloc[:, 25:], axis = 1) #assign columns to wavelength range to consider
        lil = np.min(df.iloc[:, 25:], axis = 1) #assign columns to wavelength range to consider
    else:
        big = np.max(df, axis = 1)
        lil = np.min(df, axis = 1)
    denom = big - lil
    df_std = pd.DataFrame()
    for col in targets_spectra.columns:           #apply for all wavelength columns
        df_std[col] = (targets_spectra.loc[:,col]-lil)/ denom 
    return df_std

def assign_targets_list(group = "all"):
    if group == 'kelp':
        target_classes = ['ecklonia', 'cystophora', 'macrocystis', 'carpophyllum','gravel', 'grass', 'rock', 'durvillaea', 'rhodophyte', 'undaria', 'phyllospora','sand', 'acrocarpia', 'hormosira', 'mussels']
    elif group == "all":
        target_classes = all_labels["Class"].unique()
    elif group == "browns":
        target_classes = ['ecklonia', 'cystophora', 'macrocystis', 'carpophyllum','gravel', 'grass', 'rock', 'durvillaea', 'rhodophyte', 'undaria', 'phyllospora','sand', 'acrocarpia', 'hormosira', 'mussels']
    elif group == "farm": 
        target_classes = ['macrocystis', 'carpophyllum','rhodophyte', 'undaria']
    return target_classes


#import the data being used if not already available
#all_data = pd.read_csv("all_targets_normalized550_sentinel_2_SRF.csv", index_col=0)
all_data = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\MEL_NZ_TAS_spectra.csv")

all_labels = pd.read_csv(r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\All_reflectance_labels.csv")
all_data = all_data.dropna(axis = 1, how = "any")
all_data = pd.merge(all_labels, all_data, left_on = "Class", right_on = "Class")
all_data.head()

#Define the filter parameters to chose target spectra
target_classes = assign_targets_list("all")
target_sites = ["new_zealand", "melbourne", "tasmania"]
target_setup = ["handheld"]

#apply the filters
all_targets = all_data[all_data["Class"].isin(target_classes)]
all_targets = all_targets[all_targets["setup"].isin(target_setup)]
all_targets = all_targets[all_targets["site"].isin(target_sites)]
all_targets = all_targets.reset_index().drop("index", axis = 1)
labels_columns = all_targets.select_dtypes(include = "object")
targets_spectra = all_targets.select_dtypes(include = "number")

#Normalize spectra to a range of 0 to 1
std_df = standardize_data(targets_spectra, "handheld")

#Choose to use raw or normalized data
spectra_to_use = pd.concat([labels_columns, std_df], axis = 1)

#Start a dictionary to contain the results
stats_dict = {}

#Set number of wavelength columns
spectral_set_length = len(spectra_to_use.select_dtypes(include = "number").columns)

#define the grouping from the chosen data
basic_group = spectra_to_use["Class"].unique()

#for each group, setup, site: make a dictionary entry with all of the statistics from calc_stats
for species in basic_group:
    spec_targets = spectra_to_use[spectra_to_use["Class"] == species]
    setups = spec_targets["setup"].unique()
    for setup in setups:
        spec_set_targets = spec_targets[spec_targets["setup"] == setup]
        sites = spec_set_targets["site"].unique()
        for site in sites:
            spec_set_site = spec_set_targets[spec_set_targets["site"] == site]
            spec_set_site_reduced = spec_set_site.select_dtypes(include = "number")
            name = f"{species}_{setup}_{site}" 
            
            stats_dict[name]= calc_stats(spec_set_site_reduced)

#make list of all combos of species, site, setup
options = stats_dict.keys()


#Make a lit of bad bands to be a gap when plotting
bad_bands_list = [num for num in range(753, 769)]


# Substring to search for of categories you want to plot
def plot_search_terms(search_words, mode, bad_bands_list, options = options, stats_dict = stats_dict):
    """ Plots the mean with standard deviation shaded of the search terms. Mode of either any or all to define search logic"""
    # Filtered list of options for subset using list comprehension
    filtered_list = [item for item in options if mode(word in item for word in search_words)]

    print(filtered_list)
    sns.set_palette("tab20")
    plt.clf()

    for w in filtered_list:
        # Plot with error bars
        plotting_data = pd.DataFrame(stats_dict[w])
        plotting_data.index = plotting_data.index+350
        mask = plotting_data.index.isin(bad_bands_list)
        plotting_data.loc[mask, :] = np.nan
        
        plt.plot(plotting_data.index, plotting_data['mean'], "-", label = w)  
        plt.fill_between(plotting_data.index, plotting_data['lower'], plotting_data['upper'], alpha=0.1)  # Fill between the lower and upper bounds
    plt.tight_layout(pad = 4, w_pad= 1, h_pad= 1)
    plt.title(search_words)
    plt.xlabel('Wavelength')
    plt.legend(loc = "lower right")
    plt.ylabel('Log (Min-Max transformed reflectance)')
    plt.ylim(0.01, 1)
    plt.yscale("log")
    plt.xticks([num for num in range(350, 1150, 100)])
    plot_name = rf'C:\Users\s4770224\Documents\coding\Spectral_analysis\Plots\drafts\{search_words}.png'
    plt.savefig(plot_name)
    plt.show()
    return plotting_data

search_words = ['acrocarpia', 'cystophora', 'carpophyllum', 'durvillaea', 'hormosira', 'phyllospora', 'sargassum', 'scytosiphon']
#Call the plotting function according to the desired search words, mode, bad bands
masked = plot_search_terms(search_words, any, bad_bands_list)