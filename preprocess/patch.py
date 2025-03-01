# import sys
# from geotile import GeoTile
# from PIL import Image
# import numpy as np
# import rasterio

# import glob
# import os
# from utils import extract_tile_name

# def tilling(image_dir):
#     # # image_dir = '/home/roberto/Documents
#     for file in sorted(glob.glob(image_dir+ '*.tif')):
#         gt_file = GeoTile(file)
#         # print(gt_file.meta)
#         tiles_dir = image_dir+'/'+'Patches' +"/" + extract_tile_name(file) 
#         print(tiles_dir)
#         gt_file.generate_tiles(tiles_dir,tile_x=64, tile_y=64, stride_x=64, stride_y=64, prefix='tile_')
        
        
        
        
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python tiling_script.py /path/to/your/images/")
#         sys.exit(1)
    
#     image_dir = sys.argv[1]
#     # Ensure the path ends with a slash
#     if not image_dir.endswith('/'):
#         image_dir += '/'
    
#     tilling(image_dir)

import sys
import glob
import numpy as np
from osgeo import gdal  # Ensure GDAL is installed: pip install gdal
from geotile import GeoTile  # Assuming GeoTile is a class from an external module

def extract_tile_name(file_path):
    # Function to extract the tile name from the file path
    return file_path.split('/')[-1].split('.')[0]

def replace_nan_values(file_path, nan_value=0.0001):
    dataset = gdal.Open(file_path, gdal.GA_Update)
    if dataset is None:
        print(f"Failed to open file {file_path}")
        return
    
    num_bands = dataset.RasterCount
    for i in range(1, num_bands + 1):
        band = dataset.GetRasterBand(i)
        array = band.ReadAsArray()
        
        # Replace NaN values
        array = np.nan_to_num(array, nan=nan_value)
        
        # Write the array back to the band
        band.WriteArray(array)
        
    dataset.FlushCache()

def tilling(image_dir):
    for file in sorted(glob.glob(image_dir + '*.tif')):
        # Replace NaN values in the file
        replace_nan_values(file, nan_value=0.0001)
        
        gt_file = GeoTile(file)
        tiles_dir = image_dir + '/' + 'Patches' + "/" + extract_tile_name(file)
        print(tiles_dir)
        
        # Generate tiles
        gt_file.generate_tiles(tiles_dir, tile_x=64, tile_y=64, stride_x=64, stride_y=64, prefix='tile_')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tiling_script.py /path/to/your/images/")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    # Ensure the path ends with a slash
    if not image_dir.endswith('/'):
        image_dir += '/'
    
    tilling(image_dir)
    print("Tiling complete.")