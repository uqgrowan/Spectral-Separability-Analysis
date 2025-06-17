#assigning class labels
""" Choose the input file path and the output file path at the bottom of the script to produce labels files with the different classification schemes of interest."""

import pandas as pd

def get_existing_labels(data):
    """ Extracts existing class labels from samples"""
    #print(data["Class"].unique()) # get list of existing labels
    labels = data["Class"]  #extract pd series of existing labels
    labels= pd.DataFrame(labels)  #make into dataframe
    return labels

def make_new_labels(labels: pd.DataFrame) -> pd.DataFrame:
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
    return labels

def label_the_data(data):
    """Make new label columsn for each classification scheme based 
    on existing labels. """
    
    class_labels = get_existing_labels(data)
    new_labels = make_new_labels(class_labels)
    
    return new_labels