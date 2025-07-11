#Formatting scripts
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def assign_targets_list(group = "macros",genus = None ):
    """Assign list of keep classes by group. If group is 'other', an optional genus parameter can be defined to look at a single class."""
    if group == 'kelp':
        target_classes = ['ecklonia', 'macrocystis', 'undaria']
    elif group == "all":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'sand', 'rock', 'acrocarpia', 'hormosira', 'mussels','macrocystis', 'carpophyllum', 'gravel', 'grass', 'filamentous_rhodophyte', 'undaria', 'frondose_rhodophyte', 'shell_litter', 'ulva', 'sargassum', 'barnacle_shells', 'worm_castings', 'scytosiphon','petalonia']
    elif group == "macros":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'acrocarpia', 'hormosira', 'macrocystis', 'carpophyllum', 'filamentous_rhodophyte', 'undaria', 'frondose_rhodophyte', 'ulva', 'sargassum', 'scytosiphon','petalonia']
    elif group == "browns":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'acrocarpia', 'hormosira', 'macrocystis', 'carpophyllum', 'undaria', 'sargassum', 'scytosiphon','petalonia']
    elif group == "farm": 
        target_classes = ['macrocystis', 'carpophyllum','filamentous_rhodophyte', 'undaria']
    elif group == "reds":
        target_classes = ['filamentous_rhodophyte', 'frondose_rhodophyte']
    elif group == "other": 
        target_classes = [genus]
    return target_classes

def filter_to_targets(df, filter_column,group = "macros", genus = None ):
    """Assign list of keep classes by group. If group is 'other', an optional genus parameter can be defined to look at a single class."""
    if group == 'kelp':
        target_classes = ['ecklonia', 'macrocystis', 'undaria']
    elif group == "all":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'sand', 'rock', 'acrocarpia', 'hormosira', 'mussels','macrocystis', 'carpophyllum', 'gravel', 'grass', 'filamentous_rhodophyte', 'undaria', 'frondose_rhodophyte', 'shell_litter', 'ulva', 'sargassum', 'barnacle_shells', 'worm_castings', 'scytosiphon','petalonia']
    elif group == "macros":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'acrocarpia', 'hormosira', 'macrocystis', 'carpophyllum', 'filamentous_rhodophyte', 'undaria', 'frondose_rhodophyte', 'ulva', 'sargassum', 'scytosiphon','petalonia']
    elif group == "browns":
        target_classes = ['ecklonia', 'phyllospora', 'durvillaea', 'cystophora', 'acrocarpia', 'hormosira', 'macrocystis', 'carpophyllum', 'undaria', 'sargassum', 'scytosiphon','petalonia']
    elif group == "farm": 
        target_classes = ['macrocystis', 'carpophyllum','filamentous_rhodophyte', 'undaria']
    elif group == "reds": 
        target_classes = ['frondose_rhodophyte','filamentous_rhodophyte']
    elif group == "other": 
        target_classes = [genus]
    filtered_df = df[df[filter_column].isin(target_classes)]
    return filtered_df, target_classes

def mask_bad_bands(data: pd.DataFrame, min_band, max_band, drop = False):
    bbl = [str(i) for i in range(min_band, max_band)]
    mask = data.columns.isin(bbl)
    data.loc[:, mask] = np.nan
    if drop:
        data.dropna(axis = 1, inplace = True)
    return data

def format_data (data_df: pd.DataFrame, keep_classes: list, min_band = "400", max_band = "925", fit_min = "400", fit_max = "925", text_col_count = 3):
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
    data_df = data_df[data_df["Class"].isin(keep_classes)]
    text_cols = data_df.iloc[:, :text_col_count]
    df=data_df.loc[:, min_band:max_band]
    df = df.transpose()
    scaler = StandardScaler().fit(df.loc[fit_min: fit_max, :])
    df_scaled = pd.DataFrame(scaler.transform(df))
    df_scaled.transpose().set_index(df.columns, inplace= True)
    df_scaled.columns = data_df.index
    df_scaled.index = data_df.loc[:, min_band:max_band].columns
    df_scaled = pd.concat([text_cols, df_scaled.transpose()], axis = 1)
    return df_scaled

def sort_classes(df, level = "genus"):
    """Sort classes according to predefined sort-order.
    Parameters:
    df : Dataframe of spectra with named label columns
    level = str either "genus" or "site"
    
    Returns:
    Dataframe sorted by the specified level    
    """
    if level == "genus":
        sort_order = ['macrocystis','undaria','ecklonia','cystophora','carpophyllum','phyllospora', 'durvillaea','sargassum', 'petalonia', 'scytosiphon', 'acrocarpia','ulva','hormosira','filamentous_rhodophyte', 'frondose_rhodophyte','grass','mussels','worm_castings', 'barnacle_shells', 'shell_litter','sand', 'gravel', 'rock',]
        sort_col = "Class"
    elif level == "site":
        sort_order= ["melbourne", "new_zealand", "tasmania"]
        sort_col = "site"
    else:
        print("Wrong sort level requested. Either 'genus' or 'site' accepted.")
        return

    # Make a sort column with categorical type, sort to match the sort_order
    df["sort_column"] = df[sort_col]
    df['sort_column'] = pd.Categorical(df['sort_column'], categories=sort_order, ordered=True)
    df = df.sort_values('sort_column')
    df.drop("sort_column", axis = 1)
    return df

def standardize_spectra(data_df, min_band, max_band):
    df=data_df.loc[:, min_band:max_band]
    df = df.transpose()
    scaler = StandardScaler().fit(df.loc[min_band: max_band, :])
    df_scaled = pd.DataFrame(scaler.transform(df))
    df_scaled.transpose().set_index(df.columns, inplace= True)
    df_scaled.columns = data_df.index
    df_scaled.index = data_df.loc[:, min_band:max_band].columns
    return df_scaled