import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks as fp
from scipy.signal import savgol_filter
import itertools
import preprocessing as prep

def find_zero_crossings(derivative):
    # Find where sign changes occur
    signs = np.sign(derivative)
    sign_changes = np.diff(signs) != 0
    # Get indices where crossings occur (between i and i+1)
    crossing_indices = np.where(sign_changes)[0]
    return crossing_indices


def filter_crossings_hierarchical(crossings, derivative, max_distance, method):
    """Use hierarchical clustering to group nearby crossings"""
    if len(crossings) <= 1:
        print("Only one crossing found, returning as is.")
        return crossings
    elif len(crossings) <1:
        raise ValueError("No spectral features found.")
    
    else:
        # Make 2d array for linkage calc
        points = np.column_stack([derivative.iloc[crossings].index, derivative.iloc[crossings]]) 
        
        # Drop NANs
        points = points[~np.isnan(points[:,1])]
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(points, method= method)
        clusters = fcluster(linkage_matrix, max_distance, criterion='distance')
        
        # Take the middle point from each cluster
        filtered = []
        for cluster_id in np.unique(clusters):
            cluster_crossings = points[clusters == cluster_id]
            filtered.append(cluster_crossings[len(cluster_crossings)//2]) 
        
        # Stack and sort the cluster outputs
        filtered = np.row_stack(filtered)
        filtered = np.sort(filtered, axis = 0)

        return np.array(filtered)