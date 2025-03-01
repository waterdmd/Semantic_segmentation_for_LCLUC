import argparse
import rasterio as rio
import numpy as np
import os
import glob
from utils import standardize_band, mask_value, transfor_bands
from rasterio.plot import show

# with rio.open(input_file) as src:
#     img = src.read()
#     print(img.shape)
#     print(type(img))
#     meta = src.meta
#     STD_bands=[]
    
#     for i in range(1, src.count):
        
#         band = src.read(i)
#         band =np.nan_to_num(band, nan =0.001)
#         standardized_band = standardize_band(band)
#         # print(standardized_band.shape)
#         STD_bands.append(standardized_band)
    
#     # print(f'len(STD_bands): {len(STD_bands)}')    
#     img = np.nan_to_num(img, nan =0.001)
#     NDVI = (img[3,:,:] - img[2,:,:]) / (img[3,:,:] + img[2,:,:])
#     # print(f'NDVI: {NDVI.shape}')
#     STD_bands.append(NDVI)
#     mask=np.vectorize(mask_value)(img[8,:,:])
#     # show(mask, cmap='viridis')
#     print(np.unique(mask))
#     # NDVI.append(mask)
#     STD_bands = STD_bands[:7]
#     STD_bands.append(NDVI)
#     STD_bands.append(mask)
#     print(f'len(STD_bands): {len(STD_bands)}') 
    
#     # and ensure the datatype is float since standardization changes the range
#     meta.update(count=len(STD_bands), dtype=rio.float32)
    
#     # Save the standardized bands to a new file
#     with rio.open(output_file, 'w', **meta) as dst:
#         for i, std_band in enumerate(STD_bands, start=1):
#             dst.write(std_band.astype(rio.float32), i)
    
band_coefficient = {"0":[-0.0029,1.0333],"1":[0.0014,0.9885],"2":[0.0009,1.0026],"3":[-0.0058,1.1007],"4":[-0.0001,1.0659],"5":[0.0048,1.0983]}
def process_tiff(input_file, output_file):
    with rio.open(input_file) as src:
        img = src.read()
        print(img.shape)
        print(type(img))
        meta = src.meta
        original_band_names = src.descriptions
        # if float(os.path.basename(input_file)[0:4])<= 2013:
        #     img = transfor_bands(band_coefficient, img)
            
        STD_bands = []
        
        for i in range(1, src.count + 1):
            band = src.read(i)
            # band = np.nan_to_num(band, nan=0.001)
            
            if i <= 16:  # Assuming you want to standardize the first 7 bands
                standardized_band = band
                # standardized_band = standardize_band(band)
                STD_bands.append(standardized_band)
        
        # NDVI calculation for bands 4 and 3 (indices 3 and 2 in zero-indexed Python)
        # NDVI = (img[3,:,:] - img[2,:,:]) / (img[3,:,:] + img[2,:,:])
        # STD_bands.append(NDVI)
        
        # Masking or processing the 8th band (index 7 in zero-indexed Python)
        mask = np.vectorize(mask_value)(img[16,:,:])
        STD_bands.append(mask)
        
        # Update metadata for saving
        # meta.update(count=len(STD_bands), dtype=rio.float32)
        meta.update({
            'count': len(STD_bands),
            'dtype': rio.float32,
            'descriptions': original_band_names + ('Custom Mask',)  # Update descriptions
        })
        
        
        # Save the processed bands to a new file
        with rio.open(output_file, 'w', **meta) as dst:
            for i, std_band in enumerate(STD_bands, start=1):
                dst.write(std_band.astype(rio.float32), i)


# for file in sorted(glob.glob(input_dir + '*.tif')):
#     output_dir = input_dir+'/'+'Processed'
#     if os.path.exists(output_dir):
#         pass
#     else:
#         os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, os.path.basename(file))
    
#     process_tiff(file, output_file)
#     print(f'Processed {file} and saved to {output_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TIFF files to standardize and calculate NDVI.")
    parser.add_argument("input_dir", type=str, help="Input directory containing TIFF files.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = os.path.join(input_dir, 'Processed')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for file in sorted(glob.glob(os.path.join(input_dir, '*.tif'))):
        output_file = os.path.join(output_dir, os.path.basename(file))
        process_tiff(file, output_file)
        print(f'Processed {file} and saved to {os.path.join(output_dir, os.path.basename(file))}')