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
        self.spectra_by_cat = {
            cat: material_spectra[material_spectra["Class"].isin(classes)]
            for cat, classes in self.materials.items()
        }
        self.odds_dict = odds_dict #if not None else {cat: (1, 0) for cat in self.materials.keys()}
    
    def choose_materials(self, maximum_classes = 6):
        """
        Binary random determination of presence/absence in the pixel for each material. If an odds dictionary is provided, classes will be included in the simulated pixels according to those odds. If no odds dictionary is provided, all classes will be simulated with 50-50 odds up to a user defined maximum number of classes. 
        
        
        Outputs
        present_materials (list): List of material categories present in the pixel.
        """
        
        if self.odds_dict: # is not None:
            return [cat for cat in self.materials if getrandbits(self.odds_dict[cat][0]) > self.odds_dict[cat][1]]
        else: 
            k = randint(0, maximum_classes)
            return sample(sorted(self.materials), k = k)    

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
    
    # def sim_single_class_pixels(self, single_pixel_count, start_value):
    #     """
    #     Simulate pixels with only one class and water to improve low-FPC and absent detections.
    #     """
    #     single_fpcs = pd.DataFrame(index = range(start_value, start_value + len(self.materials.keys())*single_pixel_count), columns=self.materials.keys(), dtype = "float")
    #     single_fpcs["water"] = 0.0
    #     single_endmembers = pd.DataFrame(index = range(start_value, start_value + len(self.materials.keys())*single_pixel_count), columns=self.materials.keys(), dtype = "int")
    #     single_results_list = {}
        
    #     # Change indexing to not duplicate previous indices
    #     first_index = start_value
        
    #     # For each allowed material type
    #     for cat in self.materials.keys():
    #         indices  = range(first_index, first_index + single_pixel_count) 
    #         mixed_pixels = pd.DataFrame(index = indices, columns = self.material_spectra.columns[1:], dtype="float")
            
    #         # For each desired extra pixel simulation
    #         for i in range(first_index, first_index + single_pixel_count):
                
    #             # Generate random FPC and water FPC, store both
    #             frac_perc_cover = random()
    #             single_fpcs.loc[i, cat] = frac_perc_cover
                
    #             water_fpc = (1- frac_perc_cover)
    #             single_fpcs.loc[i, "water"] = water_fpc
                
    #             # Pick endmember, store
    #             pixel_endmember = self.spectra_by_cat[cat].sample(n=1, axis = 0)
    #             single_endmembers.loc[i, cat] = pixel_endmember.index[0] 
    #             pixel_endmember = pixel_endmember.values.flatten()[1:]
                                
    #             # Pick a water endmember, store
    #             water_endmember = self.water_spectra.sample(n=1, axis = 0)
    #             single_endmembers.loc[i, "water"] = water_endmember.index[0]

    #             # Mixe pixel, store
    #             mixed_pixel = pixel_endmember * frac_perc_cover + water_endmember *water_fpc
    #             mixed_pixels.loc[i, :] = mixed_pixel.values.flatten()

    #         single_results_list[cat] = mixed_pixels
            
    #         # Advance the indexing for the next category's pixels
    #         first_index += single_pixel_count

    #     single_results_df = pd.concat((pd.DataFrame({**{'Code': key}, **value})\
    #             for key, value in single_results_list.items()), ignore_index=False)
        
    #     return single_results_df, single_fpcs, single_endmembers
    
    
    def sim_single_class_pixels(self, single_pixel_count, start_value):
        total = len(self.materials) * single_pixel_count
        idx = range(start_value, start_value + total)
        cols = list(self.materials.keys())

        single_fpcs = pd.DataFrame(0.0, index=idx, columns=cols + ["water"], dtype="float")
        single_endmembers = pd.DataFrame(0, index=idx, columns=cols,dtype = "int") 
        single_results_list = {}

        first_index = start_value

        for cat in self.materials.keys():
            indices = range(first_index, first_index + single_pixel_count)

            # Pre-sample all endmembers at once
            cat_samples = self.spectra_by_cat[cat].sample(n=single_pixel_count, replace=True)
            water_samples = self.water_spectra.sample(n=single_pixel_count, replace=True)

            # Generate all FPCs at once
            fpcs = np.random.random(single_pixel_count)
            water_fpcs = 1.0 - fpcs

            # Store FPCs
            single_fpcs.loc[indices, cat] = fpcs
            single_fpcs.loc[indices, "water"] = water_fpcs

            # Store endmember indices
            single_endmembers.loc[indices, cat] = cat_samples.index.values
            # Note: store water endmember indices if needed (add water col to single_endmembers if required)

            # Vectorised pixel mixing
            spec_cols = self.material_spectra.columns[1:]
            cat_spectra = cat_samples[spec_cols].values        # (n, bands)
            water_spectra = water_samples[spec_cols].values    # (n, bands) -- adjust col selection as needed

            mixed = cat_spectra * fpcs[:, None] + water_spectra * water_fpcs[:, None]
            mixed_pixels = pd.DataFrame(mixed, index=indices, columns=spec_cols)
            single_results_list[cat] = mixed_pixels

            first_index += single_pixel_count

        single_results_df = pd.concat(
            [df.assign(Code=key) for key, df in single_results_list.items()]
        )

        return single_results_df, single_fpcs, single_endmembers
       
    def add_water_only_pixels(self, water_pixels_count, start_value):
        " Add pure water pixels to the simulated datasets, with unique indices to avoid overlap with previous simulations."
        idx = range(start_value, start_value + water_pixels_count)
        
        water_fpcs = pd.DataFrame(0, index=idx, columns=self.materials.keys(), dtype=float)
        water_fpcs["water"] = 1.0
        
        # Sample all rows at once and re-index
        sampled = self.water_spectra.sample(n=water_pixels_count, replace=False)
        water_results_df = sampled.set_index(pd.Index(idx))
        
        water_endmembers = pd.DataFrame(np.nan, index=idx, columns=self.materials.keys(), dtype=float)
        water_endmembers["water"] = sampled.index.values
        
        return water_results_df, water_fpcs, water_endmembers
    