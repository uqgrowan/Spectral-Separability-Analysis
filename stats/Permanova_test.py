#trying Permanova

data = data_import.iloc[0:30, :]

import pandas as pd
import numpy as np
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial import distance_matrix


#Load data
data_import = pd.read_csv(r"Peak_maxima_for_permanova.csv")
data = data_import[data_import["Class"].isin(["macrocystis", "ecklonia"])]


# Extract groups
groups = data['Class']

# Extract features
features = data.drop(['Class'], axis=1)

# Calculate distance matrix
dist = pd.DataFrame(distance_matrix(x=features, y= features, threshold= 10000))

# Create DistanceMatrix with sample IDs
sample_ids = data.index
dist_mat = DistanceMatrix(dist, ids= sample_ids)

#Run permanova and show results
perma = permanova(dist_mat, grouping = groups, permutations=999)
perma
