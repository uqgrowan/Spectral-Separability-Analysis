"""
Class for pixel simulations through linear mixing
"""

from random import getrandbits, shuffle, random, sample, randint
import numpy as np
import pandas as pd


class LinearMixing:
    """
    Simulate pixels using linear mixing of endmembers according to randomly generated fractional percent covers. The presence of materials in the pixel can be determined by user defined odds, or by a random draw with a user defined maximum number of classes. The results are returned as a DataFrame with the simulated spectra, and separate DataFrames for the fractional percent covers and endmember indices used in each pixel. Additional functions allow for the simulation of single-class pixels and pure water pixels to improve low-FPC and absent detections.
    
    Args
        materials (dict): Dictionary with material categories as keys and the assigned classes lists as values.
        
        cover_ranges (dict): Dictionary with material categories as keys and tuples of (min_cover, max_cover) as values.
        
        odds_dict (dict): Dictionary of tuples with the first value being the bit size to pass to make a random value, and the second being the value the random integer must be larger than for a material to be included in a pixel. e.g. (1, 0) for 50-50 odds, (3, 5) for 25% odds of inclusion
        
        pixels (int): Number of pixels to simulate
        
        material_spectra (pd.DataFrame): DataFrame where each column represents a material's spectral signature, and a "Class" column indicates the class of each spectrum.
        
        water_spectra (pd.DataFrame): DataFrame of water spectra to be used as the background endmember, with the same structure as material_spectra.
    """
    
    def __init__(self, materials = dict, cover_ranges = list, odds_dict = dict, pixels = int, material_spectra= pd.DataFrame, water_spectra = pd.DataFrame):
        """
        Initialize the LinearMixing model.

        Args:
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
    
    def make_placeholder(self, rows, cols):
        """
        Create a placeholder DataFrame to store the results of the linear mixing model.
        
        Returns:
        pd.DataFrame: DataFrame initialized with NaN values for each material category.
        """
        placeholder = pd.DataFrame(index=rows, columns = cols, dtype=float)
        return placeholder

    def choose_materials(self):
        """
        Binary random determination of presence/absence in the pixel for each material. If an odds dictionary is provided, classes will be included in the simulated pixels according to those odds. If no odds dictionary is provided, all classes will be simulated with 50-50 odds up to a user defined maximum number of classes. 
        
        
        Outputs
        present_materials (list): List of material categories present in the pixel.
        """
        present_materials = []
        for item in self.materials.items():
            present = getrandbits(1)
            if present:
                present_materials.append(item[0])
            else:
                pass
        return present_materials

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
            spectrum = cat_spectra.sample(n=1, axis=0, replace = True)
            pixel_endmembers[cat] = spectrum
            endmember_indices[cat] = spectrum.index[0]
        return pixel_endmembers, endmember_indices

    def calculate_pixel(self, set_max_classes):
        """
        Simulate the isgnal of a  mixed pixel according to the randomly alotted fractional percent covers, and randomly selected endmembers.
        """
        # Determine materials present in pixel
        materials_list = self.choose_materials(set_max_classes)
        
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
    
    def sim_many_pixels(self, set_max_classes = False):
        """
        Simulate multiple mixed pixels and store the results in a DataFrame.
        """
        results = self.make_placeholder(rows=self.material_spectra.index, cols=range(self.pixels))
        
        for pixel in range(self.pixels):
            print(F"Simulating pixel {pixel+1}")
            mixed_pixel = self.calculate_pixel()
            results[pixel] = mixed_pixel.values.flatten()
        
        return results
        
        