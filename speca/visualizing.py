import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualizer:
    
    def __init__(self, spectra, labels, start_nm=400, end_nm=900):
        """
        Initialize a visualizer with reflectance data.
        """
        
        self.spectra = spectra
        self.labels = labels
        self.start_nm = start_nm
        self.end_nm = end_nm
        
        # Initialize variables for later use
        self.stats_dict = None
        self.spec_site_reduced = None
        
        
    def calc_stats(self, data):
        """ 
        Calculate the basic stats for each wavelength in the spectral data.
        """
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
        stats_df['wavelength'] = stats_df['wavelength'] + (self.start_nm-1)

        # Drop the first column (index column from reset_index)
        stats_df = stats_df.iloc[:, 1:]

        # Calculate lower and upper bounds for plotting
        stats_df['lower'] = stats_df['mean'] - stats_df['st_dev']
        stats_df['upper'] = stats_df['mean'] + stats_df['st_dev']
        
        return stats_df


    def class_stats(self,
                    grouping = "Class",
                    plot_by_site = True
                    ):
        """ Calculate statistics for all target classes"""

        #Start a dictionary to contain the results
        stats_dict = {}

        # Define the grouping from the chosen data
        basic_group = self.labels[grouping].unique()

        spectra_to_use = pd.concat([self.labels, self.spectra], axis = 1)
        
        # For each group, site: make stats dict
        for species in basic_group:
            spec_targets = spectra_to_use[spectra_to_use[grouping] == species]
            if plot_by_site: 
                sites = spec_targets["site"].unique()
                for site in sites:
                    spec_site = spec_targets[spec_targets["site"] == site]
                    spec_site_reduced = spec_site.select_dtypes(include= "number")
                    name = f"{species}_{site}"
                    # Calculate the stats
                    stats_dict[name]= self.calc_stats(spec_site_reduced)
            else:
                spec_reduced = spec_targets.select_dtypes(include= "number")
                # Calculate the stats
                stats_dict[species]= self.calc_stats(spec_reduced)
        self.stats_dict = stats_dict


    def plot_search_terms(self, 
                          search_words,
                          mode,
                          search_sites = None,
                          save_plot = False,
                          vbars = []):
        """ 
        Plots the mean with standard deviation shaded of the search terms.
        Optional filtering to limit by site if class stats were by site.
        Mode of either any or all to define search logic.
        """
        
        search_sites = search_sites if search_sites is not None else []
        stats_dict = self.stats_dict
        
        #make list of all combos of species, site, setup
        options = self.stats_dict.keys()

        # Filtered list of options for subset using list comprehension
        filtered_list = [item for item in options if mode(word in item for word in search_words)]
        if search_sites:
            filtered_list = [item for item in filtered_list if all(word in item for word in search_sites)]

        sns.set_palette("tab20")
        sns.set_theme(font = "Times New Roman")
        plt.clf()
        for w in filtered_list:
            # Plot with error bars
            plotting_data = pd.DataFrame(stats_dict[w])
            plotting_data.index = plotting_data.index + self.start_nm       
            plt.plot(plotting_data.index, plotting_data['mean'], "-", label = w)  
            plt.fill_between(plotting_data.index, plotting_data['lower'], plotting_data['upper'], alpha=0.2)
        plt.vlines(x= vbars, ymin = 0, ymax = 1)
        plt.tight_layout(pad = 4, w_pad= 1, h_pad= 1)
        plt.title("Mean reflectance spectra +/- st.dev")
        plt.xlabel('wavelength (nm)')
        plt.legend(loc = "upper left")
        plt.ylabel('Log(Standardized Reflectance)')
        plt.xticks([num for num in range(self.start_nm, self.end_nm +1, 100)])
        if save_plot:
            plot_name = rf'.\plots\profiles\{search_words}.svg'
            plt.savefig(plot_name)
        plt.show()
        return
