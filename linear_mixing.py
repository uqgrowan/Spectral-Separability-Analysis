import numpy as np
import pandas as pd
from random import getrandbits, shuffle, random

class LinearMixing:
    
    def __init__(self, materials = dict, cover_ranges = list, pixels = int, material_spectra= pd.DataFrame, water_spectra = pd.DataFrame):
        """
        Initialize the LinearMixing model.

        Parameters:
        materials_spectra (pd.DataFrame): DataFrame where each column represents a material's spectral signature.
        cover_ranges (dict): Dictionary with material categories as keys and tuples of (min_cover, max_cover) as values.
        materials (dict): Dictionary with material categories as keys and the assigned classes lists as values.
        pixels (int): number of pixels to simulate
        """
        self.materials = materials
        self.cover_ranges = cover_ranges
        self.pixels = pixels
        self.water_spectra = water_spectra
        self.material_spectra = material_spectra
    
<<<<<<< Updated upstream
    def make_placeholder(self, rows, cols):
=======
    def choose_materials(self, set_max_classes):
>>>>>>> Stashed changes
        """
        Create a placeholder DataFrame to store the results of the linear mixing model.
        
        Returns:
        pd.DataFrame: DataFrame initialized with NaN values for each material category.
        """
        placeholder = pd.DataFrame(index=rows, columns = cols, dtype=float)
        return placeholder

    def choose_materials(self):
        """
        For each material category specified, randomly select if that material is present in the pixel. Defaults to 50-50 odds of being in pixel.
        
        Outputs
        present_materials (list): List of material categories present in the pixel.
        """
<<<<<<< Updated upstream
        present_materials = []
        for item in self.materials.items():
            present = getrandbits(1)
            if present:
                present_materials.append(item[0])
            else:
                pass
        return present_materials
=======
        if set_max_classes is False:
            return [cat for cat in self.materials if getrandbits(self.odds_dict[cat][0]) > self.odds_dict[cat][1]]
        else: 
            k = randint(1, set_max_classes)
            return sample(sorted(self.materials), k = k)   

        
        # if self.odds_dict: # is not None:
        #     return [cat for cat in self.materials if getrandbits(self.odds_dict[cat][0]) > self.odds_dict[cat][1]]
        # else: 
        #     k = randint(1, maximum_classes)
        #     return sample(sorted(self.materials), k = k)    
>>>>>>> Stashed changes

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
            fpc = random(min_cover*surface, max_cover*surface)
            surface -= fpc
            fpc_dict[cat] = fpc
        return fpc_dict

    def pick_spectra(self, present_materials):
        """
        Randomly select a spectral signature for each material present in the pixel.
        """
        pixel_endmembers = {}
        for cat in present_materials:
            cat_spectra = self.material_spectra[self.material_spectra["Class"].isin(self.materials[cat])]
            spectrum = cat_spectra.sample(n=1, axis=1)
            pixel_endmembers[cat] = spectrum
        return pixel_endmembers

    def calculate_pixel(self, set_max_classes):
        """
        Simulate the isgnal of a  mixed pixel according to the randomly alotted fractional percent covers, and randomly selected endmembers.
        """
        # Determine materials present in pixel
        materials_list = self.choose_materials(set_max_classes)
        
        # Generate random FPC for each present material
        frac_perc_covers = self.generate_random_compositions(materials_list)
        
        # Pick endmembers for present materials
        pixel_endmembers = self.pick_spectra(materials_list)
        
        # Calculate water fpc and pick an endmember
        water_fpc = 1- sum(frac_perc_covers.values())
        water_endmember = self.water_spectra.sample(n=1, axis = 1)
        
        # Mix the pixel
        mixed_pixel = water_fpc * water_endmember
        for item in frac_perc_covers.items():
            fraction = item[1]
            print(F"Mixing {fraction} of {item[0]}")
            endmember = pixel_endmembers[item[0]]
            mixed_pixel = mixed_pixel +(fraction * endmember)
        
        print(F"Mixing {water_fpc} of water")
        
        return mixed_pixel
    
    def sim_many_pixels(self, set_max_classes = False):
        """
        Simulate multiple mixed pixels and store the results in a DataFrame.
        """
<<<<<<< Updated upstream
        results = self.make_placeholder(rows=self.material_spectra.index, cols=range(self.pixels))
=======
        fpcs = {}
        endmembers = {}
        results_list = []
    
        for i in range(self.pixels):
            mixed_pixel, fpc, endmember_indices = self.calculate_pixel(set_max_classes)
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
>>>>>>> Stashed changes
        
        for pixel in range(self.pixels):
            print(F"Simulating pixel {pixel+1}")
            mixed_pixel = self.calculate_pixel()
            results[pixel] = mixed_pixel.values.flatten()
        
        return results
        
        