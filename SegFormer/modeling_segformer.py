#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from seg_dataset import SegmentationDataset, SegmentationDatasetTwoMonths, SegmentationDatasetSeasonal, SegmentationDatasetRGB
import segmentation_models_pytorch as smp 
import seaborn as sns
import matplotlib.pyplot as plt
from segformer_pytorch import Segformer
import torch.nn.functional as F
import os 
import glob
from torchmetrics.classification.jaccard import MulticlassJaccardIndex as jaccard
import time
# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# %%
EPOCHS = 200
BS = 128
start_time = time.time()
# ###============================Seasonal Data=====================================
# train_ds = SegmentationDatasetSeasonal(data_path='/data/data/Seasonalimage/train/')
# # sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
# train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 16, pin_memory=True)
# val_ds = SegmentationDatasetSeasonal(data_path='/data/data/Seasonalimage/val/')
# # sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
# val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 16, pin_memory=True)



# ##============================Two Months (J+D) Data=============================
# #%% Instantiate Dataset and Dataloader
# train_ds = SegmentationDatasetTwoMonths(data_path='/data/data/J+D/2023_merged/Train')
# # sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
# train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 32, pin_memory=True)
# val_ds = SegmentationDatasetTwoMonths(data_path='/data/data/J+D/2023_merged/Test')
# # sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
# val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 32, pin_memory=True)


# ##============================Yearly median Data=============================
# train_ds = SegmentationDataset(data_path='/data/data/yearlyImage/Train')
# # sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
# train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 32, pin_memory=True)
# val_ds = SegmentationDataset(data_path='/data/data/yearlyImage/Val')
# # sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
# val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 32, pin_memory=True)

##============================RGB Yearly median Data=============================
train_ds = SegmentationDatasetRGB(data_path='/data/data/yearlyImage/Train')
# sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 32, pin_memory=True)
val_ds = SegmentationDatasetRGB(data_path='/data/data/yearlyImage/Val')
# sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 32, pin_memory=True)



###================================================================================
load_time = round(time.time() - start_time , 2)
#%%
W_train = train_ds.compute_class_weights_for_crossentropy()
W_val = val_ds.compute_class_weights_for_crossentropy()
#%%
for DD in val_dataloader:
    print(DD['image'].shape, DD['mask'].shape)
    
    break
# # print(val_ds.weights)
# print(len(val_ds.weights))
# # print(train_ds.weights)
# print(len(train_ds.weights))


# %%
#####B5
# model = Segformer(
#     dims = (32, 64, 160, 256),     # Adjusted to ensure divisibility by the number of heads
#     heads = (1, 2, 5, 8),           # Keep heads as they divide the corresponding dimensions correctly
#     ff_expansion = (4, 4, 4, 4),    # Feedforward expansion factor of each stage
#     reduction_ratio = (8, 4, 2, 1), # Reduction ratio for each stage for efficient attention
#     num_layers = (3, 6, 40, 3),     # Number of layers in each stage
#     channels = 3,                  # Number of input channels
#     decoder_dim = 64,               # Decoder dimension
#     num_classes = 9                 # Number of segmentation classes
# )

####B4
model = Segformer(
    dims = (32, 64, 160, 256),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = (3,8,27,3),                 # num layers of each stage
    channels = 3,                   # input channels
    decoder_dim = 64,              # decoder dimension
    num_classes = 9                 # number of segmentation classes
)
model.to(DEVICE)

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.001),
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

#%% chek if pretrained weights are available; if so, load them
wights = glob.glob("SegFormer*.pth")
if wights:
    model.load_state_dict(torch.load(f'SegFormer_epochs_{EPOCHS}_crossentropy_state_dict.pth'))
    print("Pretrained weights loaded")
else:
    print("No pretrained weights found, intializing random weights...")    

# %%
criterion = nn.CrossEntropyLoss(weight=W_train).to(DEVICE)
# (weight = W_train.to(DEVICE))


IoU = jaccard( num_classes= 9, average='none').to(DEVICE)
# %%
train_losses, val_losses = [],[]


# %%
Prev_loss = 10000
min_loss = 10000
for e in range(EPOCHS):
    model.train()
    running_train_loss, running_val_loss = 0, 0
    metrics = {'iou_scores': [], 'f1_scores': [], 'f2_scores': [], 'accuracies': [], 'recalls': [], 'ious': [], 'losses': []}
    for i, data in enumerate(train_dataloader):
        #training phase
        image_i, mask_i = data['image'], data['mask']
        image = image_i.to(DEVICE)
        mask = mask_i.to(DEVICE)
        
        # reset gradients
        optimizer.zero_grad() 
        #forward
        output = model(image.float())
        # Upsample the output to match the target label size
        output_upsampled = output
        # F.interpolate(output, size=mask.shape[1:], mode = 'bilinear',   align_corners=False)
        # calc losses
        train_loss = criterion(output_upsampled .float(), mask.long())

        # back propagation
        train_loss.backward()
        optimizer.step() #update weight          
        running_train_loss += train_loss.item()
        
        # adding metrics
        IoU_metric = IoU(output_upsampled .float(), mask.long())
        _, pred = torch.max(output_upsampled, 1)
        tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode='multiclass', num_classes=9)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        #storing the metrics in a dictionary
        metrics['ious'].append(IoU_metric)
        metrics['iou_scores'].append(iou_score)
        metrics['f1_scores'].append(f1_score)
        metrics['f2_scores'].append(f2_score)
        metrics['accuracies'].append(accuracy)
        metrics['recalls'].append(recall)
        metrics['losses'].append(train_loss.item())
        
    train_losses.append(running_train_loss) 
    
    # Compute mean of each metric
    mean_metrics =  {
    metric: np.mean([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values])
    for metric, values in metrics.items()
}
    print(f"Epoch: {e}, Training Mean Loss: {mean_metrics['losses']}, Mean IoU: {mean_metrics['ious']}, "
        f"Mean IoU Score: {mean_metrics['iou_scores']}, Mean F1 Score: {mean_metrics['f1_scores']}, ")
        # f"Mean F2 Score: {mean_metrics['f2_scores']}, Mean Accuracy: {mean_metrics['accuracies']}, "
        # f"Mean Recall: {mean_metrics['recalls']}")
    
    # validation
    model.eval()
    val_metrics = {'iou_scores': [], 'f1_scores': [], 'f2_scores': [], 'accuracies': [], 'recalls': [], 'ious': []}
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            image_i, mask_i = data['image'], data['mask']
            image = image_i.to(DEVICE)
            mask = mask_i.to(DEVICE)
            #forward
            output = model(image.float())
            output_upsampled = output
            
            # F.interpolate(output, size=mask.shape[1:],mode = 'bilinear', align_corners=False)
            # calc losses
            val_loss = criterion(output_upsampled.float(), mask.long())
            running_val_loss += val_loss.item()
            
            # Calculate additional metrics
            _, pred = torch.max(output_upsampled, 1)
            IoU_metric = IoU(output_upsampled.float(), mask.long())
            tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode='multiclass', num_classes=9)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
            # Store metrics in the validation metrics dictionary
            val_metrics['ious'].append(IoU_metric)
            val_metrics['iou_scores'].append(iou_score)
            val_metrics['f1_scores'].append(f1_score)
            val_metrics['f2_scores'].append(f2_score)
            val_metrics['accuracies'].append(accuracy)
            val_metrics['recalls'].append(recall)
            
    val_losses.append(running_val_loss)
    # Compute mean of each metric and loss
    mean_val_metrics = {
    metric: np.mean([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values])
    for metric, values in val_metrics.items()
}
    mean_val_loss = np.mean(running_val_loss / len(val_dataloader))
    # Append the average validation loss for this epoch
    # val_losses.append(mean_val_loss)

# Log or print validation metrics and loss
    print(f"Validation Loss: {mean_val_loss}, Mean IoU: {mean_val_metrics['ious']}, "
      f"Mean IoU Score: {mean_val_metrics['iou_scores']}, Mean F1 Score: {mean_val_metrics['f1_scores']}, ")
    #   f"Mean F2 Score: {mean_val_metrics['f2_scores']}, Mean Accuracy: {mean_val_metrics['accuracies']}, "
    #   f"Mean Recall: {mean_val_metrics['recalls']}")
    
    
    
    if np.median(running_val_loss) < min_loss:
        print(f"Loss value improved from {min_loss} to {np.median(running_val_loss)}; Saving model weights...")
        torch.save(model.state_dict(), f'SegFormer_epochs_{EPOCHS}_crossentropy_state_dict.pth')
        Prev_loss = np.median(running_val_loss)
        if min_loss > Prev_loss:
            min_loss = Prev_loss
            
                # Write report to text file
        with open('report.txt', 'a') as file:  # 'a' mode for appending in case this happens multiple times
            file.write(f"Epoch: {e}, Median Validation Loss: {running_train_loss},\n")
            file.write(f"Epoch: {e}, Median Validation Loss: {running_val_loss},\n")
            file.write("Mean Validation Metrics:\n")
            for metric, value in mean_val_metrics.items():
                file.write(f"{metric}: {value}\n")
            file.write("Metrics training Criteria (if any):\n")
            for metric, value in mean_metrics.items():
                file.write(f"{metric}: {value}\n")
           
    print(f"Epoch: {e}: Train Cumulative Loss: {np.median(running_train_loss)}, Val cumulative Loss: {np.median(running_val_loss)} ")

train_time = round(time.time() - load_time , 2)
print(f"Training time: {train_time} seconds")
#%% TRAIN LOSS
plt.figure(figsize=(10, 5))  
sns.lineplot(x = range(len(train_losses)), y= train_losses)
sns.lineplot(x = range(len(train_losses)), y= val_losses)

# Adding titles and labels
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()  # Show legend to identify the lines
plt.savefig('trainloss.png')
plt.show()  # Display the plotplt.show()

