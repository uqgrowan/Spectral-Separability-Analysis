#assigning class labels
""" Choose the input file path and the output file path at the bottom of the script to produce labels files with the different classification schemes of interest."""

"""
['ecklonia' 'phyllosphora' 'durvillaea' 'cystophora' 'sand' 'rock'
 'phyllospora' 'acrocarpia' 'hormosira' 'mussels' 'macrocystis'
 'carpophyllum' 'gravel' 'grass' 'filamentous_rhodophyte' 'undaria' 'bryozoans' 'macrocytis'
 'frondose_rhodophyte' 'shell_litter' 'ulva' 'filamentose_rhodophyte'
 'sargassum' 'barnacle_shells' 'worm_castings' 'scytosiphon' 'petalonia']
"""

import pandas as pd

path =r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\MEL_NZ_TAS_spectra.csv"
data = pd.read_csv(path)

print(data["Class"].unique()) # get list of existing labels
labels = data["Class"]  #extract pd series of existing labels
labels= pd.DataFrame(labels)  #make into dataframe


whites = ["wr", "stray_light", 'bad', 'bryozoans', 'glint', 'error', "background"]

#KNON grouping
kelp = ['ecklonia', 'macrocystis',  'undaria', 'petalonia']  #list of kelp labels
non_kelp = ['acrocarpia', 'barnacle_shells', 'carpophyllum', 'cystophora', 'durvillaea', 'filamentous_rhodophyte','frondose_rhodophyte', 'gravel', 'grass','hormosira', 'mussels','phyllospora', 'rhodophyte', 'rock',  'sand', 'sargassum', 'scytosiphon', 'shell_litter', 'rock', 'ulva', 'worm_castings']
labels.loc[labels["Class"].isin (kelp), "ko"] = "kelp" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (non_kelp), "ko"] = "non_kelp"  #assign other flora
labels.loc[labels["Class"].isin (whites), "ko"] = "wr"  #assign abiotic

#KOM grouping
kelp = ['ecklonia', 'macrocystis', 'petalonia', 'undaria']  #list of kelp labels
other = ['acrocarpia', 'carpophyllum', 'cystophora', 'durvillaea', 'filamentous_rhodophyte','frondose_rhodophyte',  'grass','hormosira', 'phyllospora', 'rhodophyte',  'sargassum',  'scytosiphon','ulva', ]
mineral = ['barnacle_shells', 'gravel', 'mussels', 'rock', 'sand', 'shell_litter', 'worm_castings', ]  #list of minerals  
labels.loc[labels["Class"].isin (kelp), "kom"] = "kelp" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (other), "kom"] = "other"  #assign other flora
labels.loc[labels["Class"].isin (mineral), "kom"] = "mineral"  #assign abiotic
labels.loc[labels["Class"].isin (whites), "kom"] = "wr"  #assign abiotic

#BOM grouping
brown_algae = ['ecklonia', 'macrocystis', 'petalonia', 'undaria', 'acrocarpia', 'cystophora', 'carpophyllum', 'durvillaea', 'hormosira', 'phyllospora', 'sargassum', 'scytosiphon']
other = ['grass', 'filamentous_rhodophyte','frondose_rhodophyte', 'rhodophyte', 'ulva'] #list of other plants and algae
mineral = ['barnacle_shells', 'gravel', 'mussels', 'rock', 'sand', 'shell_litter', 'worm_castings' ]  #list of minerals 
labels.loc[labels["Class"].isin (brown_algae), "bom"] = "brown_algae" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (other), "bom"] = "other"  #assign other flora
labels.loc[labels["Class"].isin (mineral), "bom"] = "mineral"  #assign abiotic
labels.loc[labels["Class"].isin (whites), "bom"] = "wr"  #assign abiotic

#KBRGM groupings
mineral = ['barnacle_shells', 'gravel', 'mussels', 'rock', 'sand', 'shell_litter', 'worm_castings', ]
red_veg = ['rhodophyte', 'filamentous_rhodophyte','frondose_rhodophyte',]
green_veg = ['grass', 'ulva']
kelp = ['ecklonia', 'macrocystis', 'petalonia', 'undaria']
brown_non_kelp = ['acrocarpia', 'cystophora', 'carpophyllum', 'durvillaea', 'hormosira', 'phyllospora', 'sargassum', 'scytosiphon']
labels.loc[labels["Class"].isin (green_veg), "kbrgm"] = "green_veg" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (kelp), "kbrgm"] = "kelp"  #assign other flora
labels.loc[labels["Class"].isin (mineral), "kbrgm"] = "mineral"  #assign abiotic
labels.loc[labels["Class"].isin (brown_non_kelp), "kbrgm"] = "other_brown_alg"  #assign abiotic
labels.loc[labels["Class"].isin (red_veg), "kbrgm"] = "red_veg"  #assign rhodophytes
labels.loc[labels["Class"].isin (whites), "kbrgm"] = "wr"  #assign abiotic

print(labels["kbrgm"].unique()) #check the new labels

labels.to_csv(r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\All_reflectance_labels.csv', index = False)