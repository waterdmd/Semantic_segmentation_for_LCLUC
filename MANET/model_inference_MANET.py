
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
model = smp.MAnet(
    encoder_name="resnext101_32x16d",
    encoder_weights=None,
    in_channels=3,
    classes=9,
    encoder_depth=5,
    activation='softmax'
)
model.eval()
model.to(DEVICE)

# model_path = '/data/models/J+D/MANET/MAnet_epochs_200_crossentropy_state_dict.pth'
# model_path = '/data/models/Seasonal/MANET/MAnet_epochs_200_crossentropy_state_dict.pth'
# model_path = '/data/models/yearly_mean/MANET/MAnet_epochs_200_crossentropy_state_dict.pth'
model_path = '/data/models/RGB/MANET/MAnet_epochs_200_crossentropy_state_dict.pth'
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
all_true_labels, all_pred_labels = [], []

with torch.no_grad():
    for data in test_dataloader:
        inputs, outputs = data['image'], data['mask']
        true = outputs.to(torch.float32).to(DEVICE)
        pred = model(inputs.to(DEVICE).float())
        _, predicted = torch.max(pred, 1)
        
        all_true_labels.extend(true.cpu().numpy().flatten())
        all_pred_labels.extend(predicted.cpu().numpy().flatten())


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

# # Save the results to a text file
# with open('summary.txt', 'w') as f:
#     f.write(f"Class-wise IoUs: {class_iou}\n")
#     f.write(f"Mean IoU: {mean_iou}\n")
#     f.write(f"Mean F1 Score: {mean_f1}\n")
#     f.write(f"Mean F2 Score: {mean_f2}\n")
#     f.write(f"Mean Accuracy: {mean_accuracy}\n")
#     f.write(f"Mean Recall: {mean_recall}\n")

#%% Confusion Matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd


Name = 'MANET'

# Compute confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(num_classes)))

# Normalize confusion matrix by row (true labels)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Convert to percentage
conf_matrix_normalized *= 100
# Create a custom colormap
cmap = sns.color_palette("rocket_r", as_cmap=True)
# Plot normalized confusion matrix
plt.figure(figsize=(6, 4.5))
ax = sns.heatmap(pd.DataFrame(conf_matrix_normalized, columns=[f'{i+1}' for i in range(num_classes)], 
                         index=[f'{i+1}' for i in range(num_classes)]), 
            annot=True, fmt='.2f', cmap=cmap, vmin=0, vmax=100)

# Set font properties
plt.xlabel('Predicted', fontsize=14, fontname='Times New Roman')
plt.ylabel('True', fontsize=14, fontname='Times New Roman')
plt.title(f'{Name}'
        #   {config_name[-5:]}'
          , fontsize=18, fontname='Times New Roman')

# Set ticks font properties
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontname='Times New Roman')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontname='Times New Roman')

# Adjust color bar font properties
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=14)
colorbar.ax.set_yticklabels([f'{int(i)}%' for i in colorbar.get_ticks()], fontsize=14, fontname='Times New Roman')

plt.savefig(f'normalized_confusion_matrix{Name}.png')
plt.show()
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

