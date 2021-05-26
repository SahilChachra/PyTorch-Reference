'''
This is a Training Pipeline with Saving Models based on Loss/Performance Metric.

'''

# Import relevant Libraries

import cv2
import glob
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# -----------------
'''
This example assumes that the DataLoader class will return {"image", "target"}. Both will be tensors.
'''
# ----------

'''
Train_model 

Input - Data loader, model, optimizer, Criterion(loss function) and the device
Output - Mean loss (sum of loss for current batch/batch size)

'''
def train_model(data_loader, model, optimizer, criterion, device):
    # This list will store loss after end of each Batch.
    loss_arr = []

    # Set model to appropriate device before training
    model.to(device)

    # Setting model to train mode
    model.train()

    
    for data in tqdm(data_loader, leave=True, desc='Training'):  # Shows Training in the progress bar
        inputs = data["image"]
        target = data["target"]
        
        inputs = inputs.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        
        # Forward Pass
        output = model(inputs)
        loss = criterion(output, target.view(-1, 1))
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_arr.append(loss.item())
        
        del inputs, target, output  # Deleteing data which will be not used again to free up GPU memory
    return round(sum(loss_arr)/len(loss_arr), 3)

'''
Evaluate_model 

Input - data_laoder, model and device
Output - return _actual (target) and _preds (class predicted by the model)
'''
def evaluate_model(data_loader, model, device):
    
    # Set the model to evaluation mode
    model.eval()
    
    _actual = []
    _preds = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, leave=True, desc='Evaluating'):   # Shows Evaluating in the progress bar
            inputs = data["image"]
            target = data["target"]
            
            inputs = inputs.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            
            pred = model(inputs)
            
            target = target.detach().cpu().numpy().tolist()
            pred = pred.detach().cpu().numpy().tolist()
            
            _actual.extend(target)
            _preds.extend(pred)
    
    return _actual, _preds  


'''
Select_model - Helps in selected models when we are trying out different architectures. Add more.
** This function is internally called by - get_configured_model_resnet()

Input - modelName and Pretrained
Output - return the model from torchvision model zoo
'''
def select_model(model_name, pretrained=True):
    
    if model_name.lower()=="resnext50":
        if pretrained:
            return models.resnext50_32x4d(pretrained=True)
        else:
            return models.resnext50_32x4d(pretrained=False)
    elif model_name.lower()=="resnet18":
        if pretrained:
            return models.resnet18(pretrained=True)
        else:
            return model.resent18(pretrained=False)
    
    elif model_name.lower()=="resnet34":
        if pretrained:
            return models.resnet34(pretrained=True)
        else:
            return model.resent34(pretrained=False)
    
    elif model_name.lower()=="resnet50":
        if pretrained:
            return models.resnet50(pretrained=True)
        else:
            return models.resent50(pretrained=False)

'''
get_configured_model_resnet - If you want to configure the RESNET model - add or remove layers, you can define a 
function to make it look clean and easy!

Input - model_name, pretrained
Output - Modified model
'''
def get_configured_model_resnet(model_name, pretrained=True):
    model = select_model(model_name, pretrained)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=1)
    )
    
    return model

'''
get_configured_model_resnet - If you want to configure the RESNEXT model - add or remove layers, you can define a 
function to make it look clean and easy!

Input - model_name, pretrained
Output - Modified model
'''
def get_configured_model_resnext(model_name, pretrained=True):
    model = select_model(model_name, pretrained)
    model.fc = nn.Linear(in_features=2048, out_features=1)
    return model


'''
This plot function plots the loss value curated after each epoch. The loss which is sent to this function is list of mean loss values
'''
def plot_training_details(epoch_loss, lr, batch_size, epochs):
    
    plt.plot(epoch_loss)
    plt.xlabel('Loss')
    plt.ylabel('Epochs')
    plt.title('Loss vs Epoch')
    
    print("Batch size {0}, LR = {1} and Epochs = {2}".format(batch_size, lr, epochs))

'''
This function is called to save model checkpoint
'''
def saveCheckpoint(state, filename):
    print("Saving Checkpoint...")
    torch.save(state, filename)
# --------------- Now here we use all the functions defined --------------------
images = train_labels.img_path.values  # Train_labels is a csv file which has columns such as img_paths, target
targets = train_labels.target.values

aug = None  # Add augmentation to DataLoader
batch_size = 64 # Normally ranges between 32, 64 and 128.

train_imgs, val_imgs, train_targets, val_targets = train_test_split(images, targets, 
                                                                    stratify=targets,
                                                                    random_state=42)

train_dataset = YourDataSetClass(img_paths=train_imgs,
                                     targets=train_targets,
                                     resize=(224, 224),
                                     augmentations=aug)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=4)

val_dataset = YourDataSetClass(img_paths=val_imgs,
                                     targets=val_targets,
                                     resize=(224, 224),
                                     augmentations=aug)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=4)

# Configure model
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
loss_fn = nn.BCEWithLogitsLoss()
lr = 0.005
model_name = "resnext50"


model = get_configured_model_resnext(model_name = model_name, pretrained=True)
# Configure Input layer as required
model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

save_model_loss = True  # Models with lower loss will be saved.
save_model_metric = False  # Models with higher performance metric will be saved (used roc auc for example)
# -- Make sure Only one them is True

min_loss = 999 # Holds initial loss value
max_roc_auc = 0

epoch_loss = []
for epoch in range(epochs):
    mean_loss_batch = train_model(train_loader, model, optimizer, loss_fn, device)
    epoch_loss.append(mean_loss_batch)
    print("Epoch : {0}, Current Mean Loss : {1}".format(epoch+1, mean_loss_batch))
    
    if save_model_loss:
        if epoch==0:
            min_loss = mean_loss_batch  # Initialize min_loss with loss after 1st epoch

        if mean_loss_batch<min_loss:
            min_loss = mean_loss_batch
            print('Current Loss is lower. Saving model...')
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
            saveCheckpoint(checkpoint, filename="{0}_loss_{1}_epoch_{2}".format(model_name, min_loss, epoch))
    
    if save_model_metric:
        actual, pred = evaluate_model(test_loader, model, device="cuda")
        curr_roc_value = metrics.roc_auc_score(actual, pred)  # ROC AUC Performance Metric for example
        if epoch==0:
            max_roc_auc = curr_roc_value  # Initialize min_loss with loss after 1st epoch

        if curr_roc_value > max_roc_auc:
            max_roc_auc = curr_roc_value
            print('Current Loss is lower. Saving model...')
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
            saveCheckpoint(checkpoint, filename="{0}_rocauc_{1}_epoch_{2}".format(model_name, max_roc_auc, epoch))

_actual, _preds = evaluate_model(train_loader, model, device)  # You can evaluate the model after each epoch also!
roc_val = metrics.roc_auc_score(_actual, _preds)
print("ROC AUC value is : ", roc_val)

plot_training_details(epoch_loss, lr, batch_size, epochs)