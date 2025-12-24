import numpy as np
import pandas as pd
from random import getrandbits, shuffle, random

class LinearMixing:
    
    def __init__(self, materials = dict, cover_ranges = list, odds_dict = dict, pixels = int, material_spectra= pd.DataFrame, water_spectra = pd.DataFrame):
        """
        Initialize the LinearMixing model.

        Parameters:
        materials_spectra (pd.DataFrame): DataFrame where each column represents a material's spectral signature.
        cover_ranges (dict): Dictionary with material categories as keys and tuples of (min_cover, max_cover) as values.
        odds_dict (dict): Dictionary of tuples with the first value being the bit size to pass to make a random value, and the second being the value the random integer must be larger than for a material to be included in a pixel. e.g. (1, 0) for 50-50 odds, (3, 5) for 25% odds of inclusion
        materials (dict): Dictionary with material categories as keys and the assigned classes lists as values.
        pixels (int): number of pixels to simulate
        """
        self.materials = materials
        self.cover_ranges = cover_ranges
        self.pixels = pixels
        self.water_spectra = water_spectra
        self.material_spectra = material_spectra
        self.spectra_by_cat = {
            cat: material_spectra[material_spectra["Class"].isin(classes)]
            for cat, classes in self.materials.items()
        }
        self.odds_dict = odds_dict if not None else {cat: (1, 0) for cat in self.materials.keys()}
    
    def choose_materials(self):
        """
        Binary random determination of presence/absence in the pixel for each material (default : 50-50 odds).
        
        
        Outputs
        present_materials (list): List of material categories present in the pixel.
        """
        
        return [cat for cat in self.materials if getrandbits(self.odds_dict[cat][0]) > self.odds_dict[cat][1]]

    def generate_random_compositions(self, present_materials):
        """
        Generate a random fractional cover for each material specified as being present in the pixel.
        
        Args
        present_materials (list): List of material categories present in the pixel.
        
        Outputs
        fpc_dict (dict): Dictionary with material categories as keys and their randomly generated fractional percent covers as values.
        """        
        fpc_dict = {}
        surface = 1
        shuffle(present_materials)
        for cat in present_materials:
            min_cover, max_cover = self.cover_ranges[cat]
            fpc = random()*(max_cover-min_cover)*surface + min_cover* surface
            surface -= fpc
            fpc_dict[cat] = fpc
        return fpc_dict

    def pick_spectra(self, present_materials):
        """
        Randomly select a spectral signature for each material present in the pixel.
        """
        pixel_endmembers = {}
        endmember_indices = {}
        for cat in present_materials:
            cat_spectra = self.spectra_by_cat[cat]
            spectrum = cat_spectra.sample(n=1, axis=0)
            pixel_endmembers[cat] = spectrum
            endmember_indices[cat] = spectrum.index[0]
        return pixel_endmembers, endmember_indices

    def calculate_pixel(self):
        """
        Simulate the isgnal of a  mixed pixel according to the randomly alotted fractional percent covers, and randomly selected endmembers.
        """
        # Determine materials present in pixel
        materials_list = self.choose_materials()
        
        # Generate random FPC for each present material
        frac_perc_covers = self.generate_random_compositions(materials_list)
        
        # Pick endmembers for present materials
        pixel_endmembers, endmember_indices = self.pick_spectra(materials_list)
        
        # Calculate water fpc and pick an endmember
        water_fpc = 1- sum(frac_perc_covers.values())
        water_endmember = self.water_spectra.sample(n=1, axis = 0)
        endmember_indices["water"] = water_endmember.index[0]
        
        fractions = np.array(list(frac_perc_covers.values()) + [water_fpc])
        spectra = np.vstack([
            *(pixel_endmembers[cat].iloc[:, 1:].to_numpy() for cat in frac_perc_covers),
            water_endmember.to_numpy()
        ])

        mixed_pixel = (fractions[:, None] * spectra).sum(axis=0)
     
        return mixed_pixel, frac_perc_covers, endmember_indices
    
    def sim_many_pixels(self):
        """
        Simulate multiple mixed pixels and store the results in a DataFrame.
        """
        fpcs = {}
        endmembers = {}
        results_list = []
    
        for i in range(self.pixels):
            mixed_pixel, fpc, endmember_indices = self.calculate_pixel()
            results_list.append(mixed_pixel)
            fpcs[i] = fpc
            endmembers[i] = endmember_indices
    
        results = np.stack(results_list)  # Combine at the end
        return results, fpcs, endmembers
        
    def format_sim_results(self, results, fpc_dict, endmembers):
        """
        Reformat the simulation results into a DataFrame with appropriate column names.
        """
        if len(results.shape) == 3:
            results = results[:,0,:]
        results_df = pd.DataFrame(results, columns=self.material_spectra.columns[1:])
        
        covers_df = pd.DataFrame(0, index = fpc_dict.keys(), columns= self.materials.keys(), dtype = float)
        endmembers_df = pd.DataFrame(np.nan, index = fpc_dict.keys(), columns= self.materials.keys(), dtype = float)
        for pixel, components in fpc_dict.items():
            for material in components.keys():
                covers_df.loc[pixel, material] = float(components[material])
        covers_df['water'] = 1 - covers_df.sum(axis=1)
                
        for pixel, components in endmembers.items():
            for material in components.keys():
                endmembers_df.loc[pixel, material] = float(components[material])
             
        return results_df, covers_df, endmembers_df