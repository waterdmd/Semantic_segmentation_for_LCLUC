from myclass import class_list, label_list
import numpy as np
import os

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
    
    return label[0] 

band_coefficient = {"0":[-0.0029,1.0333],"1":[0.0014,0.9885],"2":[0.0009,1.0026],"3":[-0.0058,1.1007],"4":[-0.0001,1.0659],"5":[0.0048,1.0983]}
def transfor_bands(band_coefficient, img):
    for i in range(0,len(img)):
    # print(i)
    # print(str(i))
        if str(i) in (band_coefficient.keys()):
            # print(band_coefficient[str(i)][0])
            img[i,:,:] = img[i,:,:]*band_coefficient[str(i)][1]+band_coefficient[str(i)][0]
        
        
    return img



def standardize_band(band):
    """Standardize the band data to have a mean of 0 and a standard deviation of 1."""
    band_mean = np.mean(band)
    band_std = np.std(band)
    return (band - band_mean) / band_std