import numpy as np
import pandas as pd

class RFC:
    
    def __init__(self, spectra, labels, labels_col="Class"):
        """
        Initialize the random forest classifier class.
        Args:
            spectra (pd.DataFrame): Reflectance data.
            labels (pd.DataFrame): Labels for the spectra.
        """
        self.spectra = spectra
        self.labels = labels
        self.labels_col = labels_col
        
        # Placeholders for later
        self.results = None