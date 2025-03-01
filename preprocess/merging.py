''' This scripts merges all the .tiff files that start with the same characters 
(e.g. 2013) in the same folder and returns the merged file in a subfolder called
'merged' in the same directory .In order to use this script, you need to run the
following command in the terminal:
""Usage: python merge.py /path/to/your/images/ 'year1,year2,...'"; 
where years strings are the four characters that the .tiff files start with 
(e.g. 2013, 2014, 2015, etc.).For me it was 2013, 2014, 2015, 2016, 2017, 2018, 
2019, 2020, 2021, 2022. The path to the images is the path to the folder that 
contains the .tiff files.
for me 
"python merging.py /Volumes/SamanData/LCLUC/LCLUC_data_MX '2013'"
'''

# Replace 'path/to/your/tiff/files/*.tif' with the actual path to your TIFF files
# years = ['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
# year = 2013
# input_dir = '/Volumes/SamanData/LCLUC/LCLUC_data_MX'

import rasterio
from rasterio.merge import merge as rio_merge  # Renamed to avoid conflict
from rasterio.plot import show
import glob
import os
import sys

def merge_tiffs(input_dir, years):
    for year in years:
        input_files = glob.glob(f'{input_dir}' + f'/*{year}*.tif')
        src_files_to_mosaic = []

        for file in input_files:
            src = rasterio.open(file)
            src_files_to_mosaic.append(src)
            
            print(f"{file} is appended")

        # Corrected the function call to avoid recursion
        
        mosaic, out_trans = rio_merge(src_files_to_mosaic)

        out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src.crs
        })

        output_dir = os.path.join(input_dir, 'merged', f"{year}_merged.tif")
        if not os.path.exists(os.path.join(input_dir, 'merged')):
            os.makedirs(os.path.join(input_dir, 'merged'))

        with rasterio.open(output_dir, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files_to_mosaic:
            src.close()

        print(f"{year} is done")

    print("All years are done")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge.py /path/to/your/images/ 'year1,year2,...'")
        sys.exit(1)

    input_dir = sys.argv[1]
    years_input = sys.argv[2]

    # Ensure the path ends with a slash
    if not input_dir.endswith('/'):
        input_dir += '/'

    # Convert the years from a comma-separated string to a list of strings
    years = [year.strip() for year in years_input.split(',')]

    merge_tiffs(input_dir, years)
