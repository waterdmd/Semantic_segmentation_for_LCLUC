



import sys
from geotile import GeoTile
from PIL import Image
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import glob
import os
from myclass import class_list, label_list


def extract_tile_name(directory):
    # Extract the base name of the file (e.g., 'tile_0.tif')
    base_name = os.path.basename(directory)
    # Split the base name by '.' and take the first part to remove the extension (e.g., 'tile_0')
    tile_name = str(os.path.splitext(base_name)[0])
    return tile_name



def mask_value(value):
    
    class_dict = class_list[1]
    label_dict = label_list[1]
    # {
    #     'Alfalfa/Hay': [36,37],
    #     'Cotton': [2,238,239],
    #     'Pecan': [74],
    #     'Corn': [1,12,13,225,226,228,237],
    #     'Other tree crops': [56,67,68,69,70,71,72,75,76,77,204,211,212,215,219,221,223],
    #     'Other row crops': [3,4,5,6,10,11,14,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,66,
    #                     205,206,207,208,209,210,213,214,216,217,218,220,222,224,227,229,230231,232,233,234,235,236,237,240,241,242,243,244,245,246,247,248,249,250,254],
    #     'Forest': [63,64,141,142,143],
    #     'Grassland/shrublands': [59,60,152,176],
    #     'Barren/fallow': [61,65,131],
    #     'Wetlands': [87,190,195],
    #     'Water bodies': [83,111,123,124],
    #     'Developed(high)/Urban': [82],
    #     'Developed(low)': [121,122,],
    #     'Other': [58,81,88,92,112],   
    #     }
    # The `label_dict` dictionary is mapping the land cover classes to numerical labels. Each key
    # represents a land cover class (e.g., 'Alfalfa/Hay', 'Cotton') and the corresponding value is a
    # list containing the numerical label assigned to that class. This mapping is useful for
    # converting the class values to numerical labels during the masking process in the `mask_value`
    # function.
    # label_dict = {
    #     'Alfalfa/Hay': [0],
    #     'Cotton': [1],
    #     'Pecan': [2],
    #     'Corn': [3],
    #     'Other tree crops': [4],
    #     'Other row crops': [5],
    #     'Forest': [6],
    #     'Grassland/shrublands': [7],
    #     'Barren/fallow': [8],
    #     'Wetlands': [9],
    #     'Water bodies': [10],
    #     'Developed(high)/Urban': [11],
    #     'Developed(low)': [12],
    #     'Other': [13],   
    #     }
    
    def invert_class_dict(class_dict):
        inverted_dict = {}
        for key, values in class_dict.items():
            for value in values:
                inverted_dict[value] = key
        return inverted_dict
    inverted_class_dict = invert_class_dict(class_dict)
    classification = inverted_class_dict.get(value)
    
    label = label_dict.get(classification, [8])  # Default to [4] if classification not found
    
    return label[0]  # Assuming each classification maps to a single mask value

band_coefficient = {"0":[-0.0029,1.0333],"1":[0.0014,0.9885],"2":[0.0009,1.0026],"3":[-0.0058,1.1007],"4":[-0.0001,1.0659],"5":[0.0048,1.0983]}

def transfor_bands(band_coefficient, img):
    for i in range(0,len(img)):
    # print(i)
    # print(str(i))
        if str(i) in (band_coefficient.keys()):
            # print(band_coefficient[str(i)][0])
            img[i,:,:] = img[i,:,:]*band_coefficient[str(i)][1]+band_coefficient[str(i)][0]
        
        
    return img






def tilling(image_dir):
    # # image_dir = '/home/roberto/Documents
    for file in sorted(glob.glob(image_dir+ '*.tif')):
        gt_file = GeoTile(file)
        # print(gt_file.meta)
        tiles_dir = image_dir+'/'+'Tiles_UNet'+"/" + extract_tile_name(file) 
        print(tiles_dir)
        gt_file.generate_tiles(tiles_dir,tile_x=64, tile_y=64, stride_x=32, stride_y=32, prefix='water_')
        # print(os.path.basename(tiles_dir))
        # print(extract_tile_name(file))
        if int(extract_tile_name(file)[:4]) >= 2013:   
            for images in sorted(glob.glob(tiles_dir+'/'+'*.tif')):
                
                print(images)
                img = rasterio.open(images)
                img = np.array(img.read())
                
                nan_percentage = np.mean(np.isnan(img[0,:,:])) * 100
                print(nan_percentage)
                if nan_percentage> 2:
                    continue
                else:
                    print(f'input shape = {img.shape}')
                    img[8,:,:]=np.vectorize(mask_value)(img[8,:,:])
                    # computing NDVI and adding it to the image
                    # NDVI = (NIR - Red) / (NIR + Red)
                    NIR = img[3,:,:]
                    Red = img[2,:,:]
                    NDVI = (NIR - Red) / (NIR + Red)
                    NDVI = np.expand_dims(NDVI, axis=0)
                    img = np.concatenate((img, NDVI), axis=0)
                    img = np.nan_to_num(img, nan =0.001)
                    for i in np.arange(0,img.shape[0],1):
                        if i==7:
                            continue
                        elif i ==8:
                            continue
                        else:
                            min = np.min(img[i,:,:])
                            max = np.max(img[i,:,:])
                            img[i,:,:] = (img[i,:,:] - min) / (max - min)
                        
                    # img = np.nan_to_num(img, nan =0.0000001, posinf = 0.0000001, neginf = 0.0000001)
                
                    print(f'output shape = {img.shape}')
                    data = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                    data[0:7,:,:] = img[0:7,:,:]
                    data[7,:,:] = img[9,:,:]
                    # data[8:17,:,:] = img[8:17,:,:]
                    data[8,:,:] = img[8,:,:]
                    zero_mask = ~np.all(data == 0, axis =(1,2))
                    data = data[zero_mask]
                    print(f'final output shape = {data.shape}')
                    save_dir = image_dir+'/'+'Tiles_Numpay_array_UNet'+"/"+os.path.basename(tiles_dir)
                    print(save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    array_dir = save_dir +"/"+ extract_tile_name(images)
                    np.save(array_dir, data)
        elif int(extract_tile_name(file)[:4]) < 2013:
            
            for images in sorted(glob.glob(tiles_dir+'/'+'*.tif')):
                print(images)
                img = rasterio.open(images)
                img = np.array(img.read())
                nan_percentage = np.mean(np.isnan(img[0,:,:])) * 100
                print(nan_percentage)
                if nan_percentage> 2:
                    continue
                else:
                    print(f'input shape = {img.shape}')
                    img[8,:,:]=np.vectorize(mask_value)(img[8,:,:])
                    NIR = img[3,:,:]
                    Red = img[2,:,:]
                    NDVI = (NIR - Red) / (NIR + Red)
                    NDVI = np.expand_dims(NDVI, axis=0)
                    img = np.concatenate((img, NDVI), axis=0)
                    img = np.nan_to_num(img, nan =0.001)
                    img = transfor_bands(band_coefficient, img)
                    for i in np.arange(0,img.shape[0],1):
                        if i==7:
                            continue
                        elif i ==8:
                            continue
                        else:
                            min = np.min(img[i,:,:])
                            max = np.max(img[i,:,:])
                            img[i,:,:] = (img[i,:,:] - min) / (max - min)
                            
                    # img = np.nan_to_num(img, nan =0.0000001, posinf = 0.0000001, neginf = 0.0000001)
                
                    print(f'output shape = {img.shape}')
                    data = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                    data[0:7,:,:] = img[0:7,:,:]
                    data[7,:,:] = img[9,:,:]
                    # data[8:17,:,:] = img[8:17,:,:]
                    data[8,:,:] = img[8,:,:]
                    zero_mask = ~np.all(data == 0, axis =(1,2))
                    data = data[zero_mask]
                    print(f'final output shape = {data.shape}')
                    save_dir = image_dir+'/'+'Tiles_Numpay_array_UNet'+"/"+os.path.basename(tiles_dir)
                    print(save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    array_dir = save_dir +"/"+ extract_tile_name(images)
                    np.save(array_dir, data)
    








if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tiling_script.py /path/to/your/images/")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    # Ensure the path ends with a slash
    if not image_dir.endswith('/'):
        image_dir += '/'
    
    tilling(image_dir)