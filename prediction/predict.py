#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import rasterio as rio
import segmentation_models_pytorch as smp 
import seaborn as sns
import matplotlib.pyplot as plt
# from segformer_pytorch import Segformer
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification.jaccard import MulticlassJaccardIndex as jaccard
from pred_DS import PredTwoMonths
from pathlib import Path
import time
# %%
# Define and load the model 
start = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # ===========================================UNET==============
# model_path = '/data/UNET_epochs_100_crossentropy_state_dict.pth'
# model = smp.Unet(
#     encoder_name = "resnext101_32x16d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=16,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#     classes=9,                      # model output channels (number of classes in your dataset)
#     encoder_depth=5,                # number of encoder backbone stages
#     activation = 'softmax'
# )

# # ===========================================MANET==============
model_path = '/data/MAnet_epochs_200_crossentropy_state_dict.pth'
model = smp.MAnet(
    encoder_name="resnext101_32x16d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=16,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=9,                      # model output channels (number of classes in your dataset)
    encoder_depth=5,                # number of encoder backbone stages
    activation = 'softmax'
)

model.load_state_dict(torch.load(model_path
                                #  , map_location=torch.device('cpu')
                                 ))
model.eval()
model.to(DEVICE)
print("Model loaded successfully")
# %%

# Load the data and create a dataloader
data_dir = '/data/Patches'
input_data = PredTwoMonths(data_dir)
input_dataloader = DataLoader(input_data, batch_size=1, shuffle=False)
print("Data loaded successfully")
# for DD in input_dataloader:
#     print(DD['geotransform'])
#     print(DD['crs'])
# %%
predict_dir = Path('/data/Predictions_MANET')
predict_dir.mkdir(parents=True, exist_ok=True)

# %%
# Function to convert CRS to a format rasterio can understand
def convert_crs(crs):
    if isinstance(crs, dict) and 'init' in crs:
        return rio.crs.CRS.from_string(crs['init'][0])
    return crs

# Make predictions and save them as single-band TIFF files
with torch.no_grad():
    for batch in input_dataloader:
        images = batch['image']
        geotransforms = batch['geotransform']
        crss = batch['crs']
        paths = batch['path']
        
        images = images.to(DEVICE)
        # geotransforms = geotransforms.to(DEVICE)
        # crss = crss.to(DEVICE)
        # paths = paths.to(DEVICE)
        
        
        images = images.float()  # Convert to float if not already
        outputs = model(images)
        predictions = outputs.argmax(dim=1).squeeze().cpu().numpy()  # Get the predicted class

        # Ensure single instance handling
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(geotransforms, list):
            geotransforms = [geotransforms]
        if not isinstance(crss, list):
            crss = [crss]
        if not isinstance(paths, list):
            paths = [paths]

        for i in range(len(predictions)):
            pred = predictions[i]
            geotransform = geotransforms[i]
            crs = convert_crs(crss[i])
            path = paths[i]

            # Ensure path is a string
            path_str = str(path)

            # Define output path
            output_path = predict_dir / f"{Path(path_str).stem}_prediction.tif"

            # Debugging: Print details
            print(f"Saving prediction to {output_path}")
            print(f"CRS: {crs}")
            print(f"Geotransform: {geotransform}")

            # Validate prediction dimensions
            if pred.ndim == 2:
                pred = np.expand_dims(pred, axis=0)  # Ensure it has a channel dimension

            # Ensure the dtype is compatible with rasterio
            pred_dtype = pred.dtype
            if pred_dtype not in [np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float32, np.float64]:
                pred_dtype = np.float32
                pred = pred.astype(np.float32)

            # Save prediction as a single-band TIFF file
            with rio.open(
                output_path,
                'w',
                driver='GTiff',
                height=pred.shape[1],
                width=pred.shape[2],
                count=1,
                dtype=pred_dtype,
                crs=crs,
                transform=geotransform,
            ) as dst:
                dst.write(pred[0], 1)

            # Verify the output file
            with rio.open(output_path) as dst:
                print(f"Saved file CRS: {dst.crs}")
                print(f"Saved file geotransform: {dst.transform}")
                
print(f"Time taken: {time.time() - start}")
# %%
