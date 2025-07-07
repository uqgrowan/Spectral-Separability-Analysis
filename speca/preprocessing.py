import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale

class PreprocessRefl:
    """
    Class for preparing reflectance spectra for later analysis using the speca toolbox.
    Preprocessing includes importing, labelling, filtering, cropping noise and optional scaling/normalizing.
    """
    def __init__(self, data_path, bad_bands = None, scaler = "standard", target_scheme = "all", target_sites = None, target_setup = None, start_nm = 400, end_nm = 900):
        
        # Initalize analysis parameters
        self.data_path = data_path
        self.bad_bands = bad_bands if bad_bands is not None else []
        self.scaler = scaler
        self.target_scheme = target_scheme
        self.target_sites = None
        self.target_setup = None
        self.start_nm = str(start_nm)
        self.end_nm = str(end_nm)
        
        # Initialize class attributes for later use
        self.data = None
        self.labels = None
        self.spectra = None
        self.target_classes = None
        self.spectra_reduc = None
        self.spectra_filtered = None
        self.spectra_norm = None
        
    def import_data(self):
        """
        Import data from .csv file with class names in leftmost column.
        Drop any incomplete columns (na values).
        Filter and mask bad bands.
        
        Returns:
            pd.DataFrame: Dataframe of all data
            pd.DataFrame: Dataframe of labels
            pd.Dataframe: Dataframe of spectra
        """
        data_import = pd.read_csv(self.data_path)
        data_import = data_import.dropna(axis = 1, how = "any")
        mask = data_import.columns.isin(self.bad_bands)
        data_import.loc[:, mask] = np.nan
        self.data = data_import
        self.labels = pd.DataFrame(self.data.select_dtypes(include = "object"))
        self.spectra = self.data.select_dtypes(include = ["float"])


    def make_new_labels(self):
        """ Create new label columns for the defined schemes.
        
        Args:
            labels [pd.DataFrame]: Existing class labels
        Returns:
            labels[pd.DataFrame]: Dataframe with new label columns added.
        """

        # Define all the classes to be used in all schemes:
        whites = ["wr", "stray_light", 'bad', 'bryozoans',
                'glint', 'error', "background"]
        kelp = ['ecklonia', 'macrocystis', 'undaria']
        brown_algae = ['ecklonia', 'macrocystis', 'petalonia', 'undaria',
                    'acrocarpia', 'cystophora', 'carpophyllum',
                    'durvillaea', 'hormosira', 'phyllospora',
                    'sargassum', 'scytosiphon']
        brown_non_kelp = ['acrocarpia', 'cystophora', 'carpophyllum',
                        'durvillaea', 'hormosira', 'petalonia',
                        'phyllospora', 'sargassum', 'scytosiphon']
        red_veg = ['rhodophyte', 'filamentous_rhodophyte',
                'frondose_rhodophyte',]
        green_veg = ['grass', 'ulva']
        non_brown_autos = ['grass', 'filamentous_rhodophyte',
                'frondose_rhodophyte', 'rhodophyte', 'ulva']
        all_non_kelp_autos = ['acrocarpia', 'carpophyllum', 'cystophora',
                    'durvillaea', 'filamentous_rhodophyte',
                    'frondose_rhodophyte', 'petalonia', 'grass','hormosira',
                    'phyllospora', 'rhodophyte',  'sargassum',
                    'scytosiphon','ulva']
        all_non_kelp = ['acrocarpia', 'barnacle_shells', 'carpophyllum',
                    'cystophora', 'durvillaea', 'filamentous_rhodophyte',
                    'frondose_rhodophyte', 'gravel', 'grass','hormosira',
                    'mussels','petalonia','phyllospora', 'rhodophyte', 'rock',
                    'sand', 'sargassum', 'scytosiphon', 'shell_litter',
                    'rock', 'ulva', 'worm_castings']
        mineral = ['barnacle_shells', 'gravel', 'mussels', 'rock',
                'sand', 'shell_litter', 'worm_castings']
        
        # Make local variable for manipulation
        labels = self.labels.copy()
        
        #Kelp/non-kelp grouping
        labels.loc[labels["Class"].isin (kelp), "ko"] = "kelp"
        labels.loc[labels["Class"].isin (all_non_kelp), "ko"] = "non_kelp"
        labels.loc[labels["Class"].isin (whites), "ko"] = "wr"
        
        #Kelp/Other-Autos/Mineral grouping
        labels.loc[labels["Class"].isin (kelp), "kom"] = "kelp"
        labels.loc[labels["Class"].isin (all_non_kelp_autos), "kom"] = "other"
        labels.loc[labels["Class"].isin (mineral), "kom"] = "mineral"
        labels.loc[labels["Class"].isin (whites), "kom"] = "wr"

        #Browns/Other-autos/Mineral grouping
        labels.loc[labels["Class"].isin (brown_algae), "bom"] = "brown_algae"
        labels.loc[labels["Class"].isin (non_brown_autos), "bom"] = "other"
        labels.loc[labels["Class"].isin (mineral), "bom"] = "mineral"
        labels.loc[labels["Class"].isin (whites), "bom"] = "wr"

        #Kelp/Brown/Red/Green/Mineral groupings
        labels.loc[labels["Class"].isin (green_veg), "kbrgm"] = "green_veg"
        labels.loc[labels["Class"].isin (kelp), "kbrgm"] = "kelp"
        labels.loc[labels["Class"].isin (mineral), "kbrgm"] = "mineral"
        labels.loc[labels["Class"].isin (brown_non_kelp), "kbrgm"] = "other_brown_alg"
        labels.loc[labels["Class"].isin (red_veg), "kbrgm"] = "red_veg"
        labels.loc[labels["Class"].isin (whites), "kbrgm"] = "wr"
        
        self.labels = labels


    def crop_noisy_wavelengths(self):
        self.spectra_reduc = self.spectra_filtered.loc[:, self.start_nm:self.end_nm ]


    def normalize_data (self) -> pd.DataFrame: 
        """
        Scale data using either min-max or standard scaling.
        Returns:
            pd.Dataframe: Scaled data.
            
        Raises:
            ValueError: If unrecognized scaler is chosen.
        """
        chosen_scaler = self.scaler.lower()
        if chosen_scaler == "minmax":
            self.spectra_norm = pd.DataFrame(minmax_scale(self.spectra_reduc, axis = 1))
        elif chosen_scaler == "standard":
            scaler = StandardScaler()
            self.spectra_norm = pd.DataFrame(scaler.fit_transform(self.spectra_reduc.T).T)
        elif chosen_scaler == "none":
            pass
        else:
            raise ValueError("Invalid scaler. Chose either 'minmax' or 'standard.")
        self.spectra_norm.index = self.spectra_reduc.index
        self.spectra_norm.columns = self.spectra_reduc.columns


    def assign_targets_list(self, genus = None):
        """
        Assign the list of targets to be included in the chosen target scheme.
        """
        if self.target_scheme == 'kelp':
            self.target_classes = ['ecklonia', 'macrocystis', 'undaria']
        elif self.target_scheme == "all":
            self.target_classes = self.labels["Class"].unique()
        elif self.target_scheme == "browns":
            self.target_classes = ['ecklonia', 'cystophora', 'macrocystis', 'carpophyllum','durvillaea', 'undaria', 'phyllospora', 'acrocarpia', 'hormosira', 'petalonia', 'sargassum', 'scytosiphon']
        elif self.target_scheme == "farm": 
            self.target_classes = ['macrocystis', 'carpophyllum','rhodophyte', 'undaria']
        elif self.target_scheme == "all_macros":
            self.target_classes = ['acrocarpia', 'carpophyllum', 'cystophora', 'durvillaea', 'ecklonia','filamentous_rhodophyte', 'frondose_rhodophyte', 'hormosira', 'macrocystis','petalonia', 'phyllospora','sargassum','scytosiphon', 'ulva', 'undaria']
        elif self.target_scheme == "all_autos":
            self.target_classes = ['acrocarpia', 'carpophyllum', 'cystophora', 'durvillaea', 'ecklonia','filamentous_rhodophyte', 'grass','frondose_rhodophyte', 'hormosira', 'macrocystis','petalonia', 'phyllospora','sargassum','scytosiphon', 'ulva', 'undaria']
        elif self.target_scheme == "reds":
            self.target_classes = ['filamentous_rhodophyte', 'frondose_rhodophyte']
        elif self.target_scheme == "other":
            self.target_classes = [genus]
        else:
            raise ValueError("Invalid target scheme. Choose from 'kelp', 'all', 'browns', 'all_macros', 'all_autos' or 'farm'.")


    def filter_targets(self):
        # Make list of target classes to keep
        self.assign_targets_list()

        # Make the mask of chosen conditions
        mask = self.labels["Class"].isin(self.target_classes)
        if self.target_setup:
            mask &= self.labels["setup"].isin(self.target_setup)
        if self.target_sites:
            mask &= self.labels["sites"].isin(self.target_sites)
        
        # Apply the filters
        self.spectra_filtered = self.spectra.loc[mask]
        self.labels_filtered = self.labels.loc[mask]
        
        # Validation
        if len(self.spectra_filtered) == 0:
            raise ValueError("No spectra remaining after filtering.")

        print(f"Data filtered from {len(self.spectra)} to {len(self.spectra_filtered)} spectra.")


    def default_chain(self):
        """
        Run preprocessing with default parameters.
        Returns:
            pd.DataFrame: Processed spectra data.
            pd.DataFrame: Processed labels data.
        """
        self.import_data()
        self.make_new_labels()
        self.filter_targets()
        self.crop_noisy_wavelengths()
        self.normalize_data()
        