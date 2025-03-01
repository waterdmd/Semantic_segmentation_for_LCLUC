
import rasterio
import numpy as np
from tiling import mask_value, transfor_bands, band_coefficient
from geotile import GeoTile







def read(image, year):
    ''' read is a function that reads a .tif file and returns a numpay array that is ready to be fed ti the model for 
    prediction. It also returns the crs of the image.'''
    
    
    
    img = rasterio.open(image)
    info = img.meta
    img = np.array(img.read())
    img[8,:,:]=np.vectorize(mask_value)(img[8,:,:])
    NIR = img[3,:,:]
    Red = img[4,:,:]
    NDVI = (NIR - Red) / (NIR + Red)
    NDVI = np.expand_dims(NDVI, axis=0)
    img = np.concatenate((img, NDVI), axis=0)
    img = np.nan_to_num(img, nan =0.001)
    img = transfor_bands(band_coefficient, img) if year< 2013 else img
    for i in np.arange(0,img.shape[0],1):
                        if i==7:
                            continue
                        elif i ==8:
                            continue
                        else:
                            min = np.min(img[i,:,:])
                            max = np.max(img[i,:,:])
                            img[i,:,:] = (img[i,:,:] - min) / (max - min)
    data = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    data[0:7,:,:] = img[0:7,:,:]
    data[7,:,:] = img[9,:,:]
                    # data[8:17,:,:] = img[8:17,:,:]
    data[8,:,:] = img[8,:,:]
    zero_mask = ~np.all(data == 0, axis =(1,2))
    data = data[zero_mask]
    data = np.moveaxis(data, 0, -1)
    print(f'final output shape = {data.shape}')
    
    return data, info

read('/Volumes/SamanData/LCLUC/LCLUC_train/Tornillo/Tiles_UNet/2008_data/tile_20.tif', 2008)

