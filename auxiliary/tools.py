import pandas as pd


def calc_class_averages(data, labels, labels_col, average_by_site = False):
    """
    Calculate class averages for the unique labels in the dataset.
    
    Raises:
        ValueError: If the number of rows in data and labels do not match.
    """
    
    if data.shape[0] != labels.shape[0]:
        raise ValueError("The number of rows in data and labels must match.")
    
    if average_by_site is True:
        select_labels = pd.DataFrame(labels.apply(lambda x: f"{x[labels_col]}-{x['site']}", axis=1))
        select_labels.columns = ["class-site"]
        combined = pd.concat([select_labels, data], axis=1)
        reduced = pd.pivot_table(combined,
                                index = ["class-site"],
                                values = combined.columns[1:],
                                aggfunc = 'mean',
                                )
    else:
        select_labels = labels[labels_col]
        combined = pd.concat([select_labels, data], axis=1)
        reduced = pd.pivot_table(combined,
                                index = labels_col,
                                values = combined.columns[1:],
                                aggfunc = 'mean',
                                )
    return reduced

def sort_classes(spectra, labels, labels_col = "Class", sort_order = None): 
    """
    Sort the classes into a specified order, or use the default if none provided.
    
    Args: 
        spectra (pd.Dataframe) : Reflectance data to be sorted
        labels (pd.DataFrame): labels to be sorted
        lables_col (str): Name of the column of labelling to be used to sort
        sort_order (list, optional): List of classes in the desired order
        
    Returns:
        pd.DataFrame: Sorted spectra
        pd.DataFrame: Sorted labels
    
    Raises:
        ValueError: If the number of rows in spectra and labels do not match.
    """
    
    if spectra.shape[0] != labels.shape[0]:
        raise ValueError("The number of rows in spectra and labels must match.")
    
    default_order = ['undaria',
                     'macrocystis',
                     'ecklonia',
                     'durvillaea',
                     'phyllospora',
                     'cystophora',
                     'carpophyllum',
                     'scytosiphon',
                     'petalonia',
                     'acrocarpia',
                     'hormosira',
                     'sargassum',
                     'frondose_rhodophyte',
                     'filamentous_rhodophyte',
                     'ulva',
                     'grass',
                     'mussels',
                     'worm_castings',
                     'barnacle_shells',
                     'shell_litter',
                     'sand',
                     'gravel',
                     'rock']
    if not sort_order:
        sort_order = default_order
    
    wavelengths = spectra.columns
    label_groups = labels.columns
    
    data = pd.concat([labels, spectra], axis = 1)
    data["sort_column"] = data[labels_col]
    data["sort_column"] = pd.Categorical(data["sort_column"], categories=sort_order, ordered=True)
    data = data.sort_values("sort_column")
    labels = data[label_groups] #select_dtypes(include="object")
    spectra = data[wavelengths] #.select_dtypes(include="number")
    return spectra, labels

def make_compound_labels(labels, columns):
    """
    Make compound labels from two existing columns.
    """
    compound_labels = labels.apply(lambda x: f"{x[columns[0]]}-{x[columns[1]]}", axis=1)
    return compound_labels