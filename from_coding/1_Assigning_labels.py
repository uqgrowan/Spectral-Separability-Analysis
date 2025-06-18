#assigning class labels
""" 
Choose the input file path and the output file path at the bottom of the script to produce labels files with the different classification schemes of interest.
"""

import pandas as pd

path =r"C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\MEL_NZ_TAS_spectra.csv"
data = pd.read_csv(path)
labels = pd.DataFrame(data["Class"])  #extract pd series of existing labels

#KNON grouping
kelp = ['ecklonia', 'macrocystis',  'undaria']  #list of kelp labels
non_kelp = ['acrocarpia', 'barnacle_shells', 'carpophyllum', 'cystophora', 'durvillaea', 'filamentous_rhodophyte','frondose_rhodophyte', 'gravel', 'grass','hormosira', 'mussels','phyllospora', 'rhodophyte', 'rock',  'sand', 'sargassum', 'scytosiphon', 'shell_litter', 'rock', 'ulva', 'worm_castings','petalonia']
labels.loc[labels["Class"].isin (kelp), "ko"] = "kelp" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (non_kelp), "ko"] = "non_kelp"  #assign other flora

#KOM grouping
kelp = ['ecklonia', 'macrocystis', 'undaria']  #list of kelp labels
other = ['acrocarpia', 'carpophyllum', 'cystophora', 'durvillaea', 'filamentous_rhodophyte','frondose_rhodophyte',  'grass','hormosira', 'phyllospora', 'rhodophyte',  'sargassum',  'scytosiphon','ulva', 'petalonia',]
mineral = ['barnacle_shells', 'gravel', 'mussels', 'rock', 'sand', 'shell_litter', 'worm_castings', ]  #list of minerals  
labels.loc[labels["Class"].isin (kelp), "kom"] = "kelp" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (other), "kom"] = "other"  #assign other flora
labels.loc[labels["Class"].isin (mineral), "kom"] = "mineral"  #assign abiotic

#BOM grouping
brown_algae = ['ecklonia', 'macrocystis', 'petalonia', 'undaria', 'acrocarpia', 'cystophora', 'carpophyllum', 'durvillaea', 'hormosira', 'phyllospora', 'sargassum', 'scytosiphon']
other = ['grass', 'filamentous_rhodophyte','frondose_rhodophyte', 'rhodophyte', 'ulva'] #list of other plants and algae
mineral = ['barnacle_shells', 'gravel', 'mussels', 'rock', 'sand', 'shell_litter', 'worm_castings' ]  #list of minerals 
labels.loc[labels["Class"].isin (brown_algae), "bom"] = "brown_algae" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (other), "bom"] = "other_autotrophs"  #assign other flora
labels.loc[labels["Class"].isin (mineral), "bom"] = "mineral"  #assign abiotic

#order grouping
laminiarales = ['ecklonia', 'macrocystis',  'undaria']  #list of kelp labels
fucales = ['durvillaea', 'acrocarpia', 'cystophora', 'hormosira', 'phyllospora', 'sargassum', 'carpophyllum']
ectocarpales = ['scytosiphon', 'petalonia']
mineral = ['barnacle_shells', 'gravel', 'mussels', 'rock', 'sand', 'shell_litter', 'worm_castings']
autotrophs = ['filamentous_rhodophyte','frondose_rhodophyte', 'grass', 'rhodophyte', 'ulva']
labels.loc[labels["Class"].isin (laminiarales), "order"] = "laminiarales" #assign all kelp
labels.loc[labels["Class"].isin (fucales), "order"] = "fucales"  #assign other flora
labels.loc[labels["Class"].isin (ectocarpales), "order"] = "ectocarpales" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (autotrophs), "order"] = "autotrophs"  #assign other flora
labels.loc[labels["Class"].isin (mineral), "order"] = "mineral"  #assign abiotic

# Colour grouping
mineral = ['barnacle_shells', 'gravel', 'mussels', 'rock', 'sand', 'shell_litter', 'worm_castings', ]
red = ['rhodophyte', 'filamentous_rhodophyte','frondose_rhodophyte',]
green = ['grass', 'ulva']
brown = ['acrocarpia','ecklonia', 'macrocystis', 'undaria', 'cystophora', 'carpophyllum', 'durvillaea', 'hormosira', 'phyllospora', 'sargassum', 'scytosiphon', 'petalonia',]
labels.loc[labels["Class"].isin (green), "colour"] = "green" #assign all kelp classes to kelp label
labels.loc[labels["Class"].isin (mineral), "colour"] = "mineral"  #assign abiotic
labels.loc[labels["Class"].isin (brown), "colour"] = "brown"  #assign abiotic
labels.loc[labels["Class"].isin (red), "colour"] = "red"  #assign rhodophytes

labels.head()
print(labels["order"].unique()) #check the new labels

labels.to_csv(r'C:\Users\s4770224\Documents\coding\Spectral_analysis\Combined_analysis\New_reflectance_labels.csv', index=False)