#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from seg_dataset import SegmentationDataset, SegmentationDatasetTwoMonths
import segmentation_models_pytorch as smp 
import seaborn as sns
import matplotlib.pyplot as plt
# from segformer_pytorch import Segformer
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification.jaccard import MulticlassJaccardIndex as jaccard

from yacs.config import CfgNode as CN

from HR_config import MODEL_CONFIGS
from HRnet import get_hrnet_model
import numpy as np
import torch
from torch.utils.data import DataLoader
from seg_dataset import SegmentationDatasetTwoMonths, SegmentationDataset, SegmentationDatasetSeasonal,SegmentationDatasetRGB,SegmentationDatasetFusion
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

#%% Dataset and Dataloader
#JD: I have changed the data path to the merged data
# test_dir = '/data/data/J+D/2023_merged/Test'
# test_ds = SegmentationDatasetTwoMonths(data_path=test_dir)
#seasonal
# test_dir = '/data/data/Seasonalimage/val/'
# test_ds = SegmentationDatasetSeasonal(data_path=test_dir)
#yearly
# test_dir = '/data/data/yearlyImage/Val'
# test_ds = SegmentationDataset(data_path=test_dir)

#RGB 
test_dir = '/data/data/yearlyImage/Val'
test_ds = SegmentationDatasetRGB(data_path=test_dir)

test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

# %% Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#%% Model setup

def load_config(config_name):
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.EXTRA = MODEL_CONFIGS[config_name]
    cfg.MODEL.ALIGN_CORNERS = True
    cfg.MODEL.INIT_WEIGHTS = True
    cfg.MODEL.PRETRAINED = ''  # Provide the correct path or leave empty if not available
    cfg.MODEL.OCR = CN()
    cfg.MODEL.OCR.MID_CHANNELS = 512
    cfg.MODEL.OCR.KEY_CHANNELS = 256
    cfg.DATASET = CN()
    cfg.DATASET.NUM_CLASSES = 9  # Update this number according to your dataset
    return cfg

config_name = 'hrnet18'  # Change to 'hrnet18' or 'hrnet32' if needed
input_channels = 3  # Update this to 8, 16, or 24 based on your dataset

# Load configuration
cfg = load_config(config_name)

# Initialize model
model = get_hrnet_model(cfg, input_channels)
model.eval()
model.to(DEVICE)

# model_path = '/data/models/J+D/HRNET/18/HRNET_epochs_200_crossentropy_state_dict.pth'
# model_path = '/data/models/Seasonal/HRNet/18/HRNET_epochs_200_crossentropy_state_dict.pth'
# model_path = '/data/models/yearly_mean/HRNET/18/HRNET_epochs_100_crossentropy_state_dict.pth'
model_path = '/data/models/RGB/HRNET/18/HRNET_epochs_200_crossentropy_state_dict.pth'
#%% Load weights
EPOCHS = 200

model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))

# %% Helper functions to calculate metrics
def calculate_metrics(tp, fp, fn, tn):
    epsilon = 1e-7
    f1 = 2 * tp / (2 * tp + fp + fn + epsilon)
    f2 = 5 * tp / (5 * tp + 4 * fn + fp + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    recall = tp / (tp + fn + epsilon)
    return f1, f2, accuracy, recall

# %% Model Evaluation
num_classes = 9
tp = np.zeros(num_classes)
fp = np.zeros(num_classes)
fn = np.zeros(num_classes)
tn = np.zeros(num_classes)  # Added true negatives

with torch.no_grad():
    for data in test_dataloader:
        inputs, outputs = data['image'], data['mask']
        true = outputs.to(torch.float32).to(DEVICE)
        pred = model(inputs.to(DEVICE).float())
        _, predicted = torch.max(pred, 1)

        for cls in range(num_classes):
            tp[cls] += torch.sum((predicted == cls) & (true == cls)).item()
            fp[cls] += torch.sum((predicted == cls) & (true != cls)).item()
            fn[cls] += torch.sum((predicted != cls) & (true == cls)).item()
            tn[cls] += torch.sum((predicted != cls) & (true != cls)).item()  # Correctly count true negatives

# Compute IoU for each class
class_iou = tp / (tp + fp + fn + 1e-7)
mean_iou = np.mean(class_iou)

# Compute additional metrics
f1_scores, f2_scores, accuracies, recalls = calculate_metrics(tp, fp, fn, tn)
mean_f1 = np.mean(f1_scores)
mean_f2 = np.mean(f2_scores)
mean_accuracy = np.mean(accuracies)
mean_recall = np.mean(recalls)

print(f"Class-wise IoUs: {class_iou}")
print(f"Mean IoU: {mean_iou}")
print(f"Mean F1 Score: {mean_f1}")
print(f"Mean F2 Score: {mean_f2}")
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Mean Recall: {mean_recall}")

# Save the results to a text file
with open('summary.txt', 'w') as f:
    f.write(f"Class-wise IoUs: {class_iou}\n")
    f.write(f"Mean IoU: {mean_iou}\n")
    f.write(f"Mean F1 Score: {mean_f1}\n")
    f.write(f"Mean F2 Score: {mean_f2}\n")
    f.write(f"Mean Accuracy: {mean_accuracy}\n")
    f.write(f"Mean Recall: {mean_recall}\n")

# #%% Median Accuracy
# pixel_accuracies = [np.mean((true == predicted).cpu().numpy()) for true, predicted in zip(outputs, pred)]
# print(f"Median Pixel Accuracy: {np.median(pixel_accuracies) * 100}")
# print(f"Median IoU: {np.median(class_iou) * 100}")

#%% Pick a test image and show it
Sample = next(iter(test_dataloader))
image_test, mask = Sample['image'], Sample['mask']
plt.imshow(np.transpose(image_test[0, 0:3, :, :].cpu().numpy(), (1, 2, 0)))

# #%% EVALUATE MODEL
# # create preds
# with torch.no_grad():
#     image_test = image_test.float().to(DEVICE)
#     output = model(image_test)

# #%%
# output_cpu = output.cpu().squeeze().numpy()
# Output = output_cpu[:,:,:]
# output_cpu = Output.transpose((1, 2, 0))
# output_cpu = output_cpu.argmax(axis=2)

# # %%
# fig, axs = plt.subplots(nrows=1, ncols=2)
# fig.suptitle('True and Predicted Mask')
# axs[0].imshow(mask[0, :, :])
# axs[1].imshow(output_cpu)
# axs[0].set_title("True Mask")
# axs[1].set_title("Predicted Mask")
# plt.savefig('MAnet_Predicted_Mask.png')
# plt.show()

# # %%
# print(np.unique(output_cpu))
# print(np.unique(mask[0, :, :].numpy()))

