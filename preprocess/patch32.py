import sys
from geotile import GeoTile
from PIL import Image
import numpy as np
import rasterio

import glob
import os
from utils import extract_tile_name

def tilling(image_dir):
    # # image_dir = '/home/roberto/Documents
    for file in sorted(glob.glob(image_dir+ '*.tif')):
        gt_file = GeoTile(file)
        # print(gt_file.meta)
        tiles_dir = image_dir+'/'+'Patches32' +"/" + extract_tile_name(file) 
        print(tiles_dir)
        gt_file.generate_tiles(tiles_dir,tile_x=32, tile_y=32, stride_x=32, stride_y=32, prefix='tile_')
        
        
        
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tiling_script.py /path/to/your/images/")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    # Ensure the path ends with a slash
    if not image_dir.endswith('/'):
        image_dir += '/'
    
    tilling(image_dir)