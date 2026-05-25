import glob
import os
import pandas as pd
import rasterio


def get_img_list(img_dir:str) -> list:
    """When pointed to a directory with raw sony agrowing jpegs, makes a list of all images to be processed.
    
    Args: 
        directory (str): Relative path to the directory with .jpegs
    
    Returns:
        dict: Dictionary with RSR filenames as keys and RSR functions as Dataframe values.
    """
    
    img_names = []
    img_files = []
    for file in glob.glob(os.path.join(img_dir, "*.tif")):
        #remove the .csv extension
        filename = os.path.basename(file)[:-4]
        img_names.append(filename)
        img_files.append(file)
    return img_files, img_names

input_dir = 'C:/Users/s4770224/Documents/Work/geospatial_analysis/water_sampling/Enmap/water_images'

files, names = get_img_list(input_dir)
len(files)

pixels = pd.DataFrame()

for file in files:
    im = rasterio.open(file)
    im_arr = im.read()
    im.close()
    im_mask = im_arr.sum(axis=0) != 0 
    image = im_arr[:, im_mask]
    image = pd.DataFrame(image.T)

    print(f"Image being added is of shape: {image.shape}")
    print(f"Pixels being added to array of shape: {pixels.shape}")
    pixels = pd.concat([pixels, image], ignore_index=True)
    print(f" {pixels.shape[0]} pixels loaded.")
    
pixels.drop_duplicates(inplace = True)    
pixels.shape

pixel_sample = pixels.sample(n=100000, replace = False)
pixel_sample.shape

pixel_sample.to_csv('C:/Users/s4770224/Documents/Work/geospatial_analysis/water_sampling/Enmap/Enmap_pixel_samples.csv', index = False)
