#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from seg_dataset import SegmentationDataset, SegmentationDatasetTwoMonths, SegmentationDatasetSeasonal, SegmentationDatasetRGB, SegmentationDatasetFusion, RandomFlipRotate
import segmentation_models_pytorch as smp 
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import os
import glob
from torchmetrics.classification.jaccard import MulticlassJaccardIndex as jaccard
# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# %%
start_time = time.time()
EPOCHS = 200
BS = 128

transform = RandomFlipRotate()
###============================Seasonal Data=====================================
# train_ds = SegmentationDatasetSeasonal(data_path='/data/data/Seasonalimage/train/', transform=transform)
# # sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
# train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 16, pin_memory=True)
# val_ds = SegmentationDatasetSeasonal(data_path='/data/data/Seasonalimage/val/', transform=transform)
# # sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
# val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 16, pin_memory=True)



# # ##============================Two Months (J+D) Data=============================
# #%% Instantiate Dataset and Dataloader
# train_ds = SegmentationDatasetTwoMonths(data_path='/data/data/J+D/2023_merged/Train', transform = transform)
# # sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
# train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 32, pin_memory=True)
# val_ds = SegmentationDatasetTwoMonths(data_path='/data/data/J+D/2023_merged/Test', transform = transform)
# # sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
# val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 32, pin_memory=True)


# ##============================Yearly median Data=============================
# train_ds = SegmentationDataset(data_path='/data/data/yearlyImage/Train')
# # sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
# train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 32, pin_memory=True)
# val_ds = SegmentationDataset(data_path='/data/data/yearlyImage/Val')
# # sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
# val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 32, pin_memory=True)

# ##============================RGB Yearly median Data=============================
# train_ds = SegmentationDatasetRGB(data_path='/data/data/yearlyImage/Train')
# # sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
# train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 32, pin_memory=True)
# val_ds = SegmentationDatasetRGB(data_path='/data/data/yearlyImage/Val')
# # sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
# val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 32, pin_memory=True)

##============================Fusion median Data=============================
train_ds = SegmentationDatasetFusion(data_path='/data/data/Fusion/Train', transform=transform)
# sampler = torch.utils.data.WeightedRandomSampler(train_ds.weights, len(train_ds.weights))
train_dataloader = DataLoader(train_ds, batch_size=BS, num_workers = 16, pin_memory=True)
val_ds = SegmentationDatasetFusion(data_path='/data/data/Fusion/Val', transform= transform)
# sampler1 = torch.utils.data.WeightedRandomSampler(val_ds.weights, len(val_ds.weights))
val_dataloader = DataLoader(val_ds, batch_size=BS,num_workers = 16, pin_memory=True)

##================================================================================
load_time = round(time.time() - start_time , 2)
print(f"Data loaded in {load_time} seconds")
#%%
W_train = train_ds.compute_class_weights_for_crossentropy()
W_val = val_ds.compute_class_weights_for_crossentropy()
#%%
for DD in train_dataloader:
    print(DD['image'].shape,  DD['mask'].shape)
    break
# print(val_ds.weights)
# print(len(val_ds.weights))
# print(train_ds.weights)
# print(len(train_ds.weights))

# %%
model = smp.MAnet(
    encoder_name="mit_b5",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=21,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=9,                      # model output channels (number of classes in your dataset)
    encoder_depth=5,                # number of encoder backbone stages
    activation = 'softmax'
)
model.to(DEVICE)

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.001),
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# %%
wights = glob.glob("MAnet*.pth")
if wights:
    model.load_state_dict(torch.load(f'MAnet_MiT_epochs_200_crossentropy_state_dict.pth'))
    print("Pretrained weights loaded")

#%%
criterion = nn.CrossEntropyLoss(weight = W_train).to(DEVICE)
IoU = jaccard( num_classes= 9, average='none').to(DEVICE)
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
        
        
        # image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False)
        # print(image.shape)
        image = image_i.to(DEVICE)
        mask = mask_i.to(DEVICE)
        
        # reset gradients
        optimizer.zero_grad() 
        #forward
        output = model(image.float())
        # Upsample the output to match the target label size
        # output = F.interpolate(output, size=mask.shape[1:], mode='bilinear', align_corners=False)
        # calc losses
        train_loss = criterion(output .float(), mask.long())

        # back propagation
        train_loss.backward()
        optimizer.step() #update weight          
        
        running_train_loss += train_loss.item()
        
        # adding metrics
        IoU_metric = IoU(output .float(), mask.long())
        _, pred = torch.max(output, 1)
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
            # output_upsampled = F.interpolate(output, size=mask.shape[1:], mode='bilinear', align_corners=False)
            # calc losses
            val_loss = criterion(output.float(), mask.long())
            running_val_loss += val_loss.item()
                        # Calculate additional metrics
            _, pred = torch.max(output, 1)
            IoU_metric = IoU(output.float(), mask.long())
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
    mean_val_metrics =  {
    metric: np.mean([v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values])
    for metric, values in val_metrics.items()
}
    mean_val_loss = np.mean(running_val_loss / len(val_dataloader))
    # Append the average validation loss for this epoch
    

# Log or print validation metrics and loss
    print(f"Validation Loss: {mean_val_loss}, Mean IoU: {mean_val_metrics['ious']}, "
      f"Mean IoU Score: {mean_val_metrics['iou_scores']}, Mean F1 Score: {mean_val_metrics['f1_scores']}, ")
    #   f"Mean F2 Score: {mean_val_metrics['f2_scores']}, Mean Accuracy: {mean_val_metrics['accuracies']}, "
    #   f"Mean Recall: {mean_val_metrics['recalls']}")
    
    
    if np.median(running_val_loss) < min_loss:
        print(f"Loss value improved from {min_loss} to {np.median(running_val_loss)}; Saving model weights...")
        torch.save(model.state_dict(), f'MAnet_MiT_epochs_{EPOCHS}_crossentropy_state_dict.pth')
        Prev_loss = np.median(running_val_loss)
        if min_loss > Prev_loss:
            min_loss = Prev_loss
            
                # Write report to text file
        with open('report.txt', 'a') as file:  # 'a' mode for appending in case this happens multiple times
            file.write(f"Epoch: {e}, Median Training Loss: {running_train_loss},\n")
            file.write(f"Epoch: {e}, Median Validation Loss: {running_val_loss},\n")
            file.write("Mean Validation Metrics:\n")
            for metric, value in mean_val_metrics.items():
                file.write(f"{metric}: {value}\n")
            file.write("Metrics training Criteria (if any):\n")
            for metric, value in mean_metrics.items():
                file.write(f"{metric}: {value}\n")
             
        
    print(f"Epoch: {e}: Train Loss: {np.mean(running_train_loss)}, Val Loss: {np.mean(running_val_loss)}")

train_time = round(time.time() -  start_time, 2)   
print(f"Training completed in {train_time} seconds for {EPOCHS} epochs")
#%% TRAIN LOSS
plt.figure(figsize=(10, 5))  
sns.lineplot(x = range(len(train_losses)), y= train_losses)
sns.lineplot(x = range(len(val_losses)), y= val_losses)

# Adding titles and labels
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()  # Show legend to identify the lines
plt.savefig('trainloss.png')
plt.show()  # Display the plot

plt.show()


# # %%
