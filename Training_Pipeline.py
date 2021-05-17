'''
This is a simple Training Pipeline for Pytorch.
All the components for training the model are defined as seperate functions.
All this can also be encapsulated in a single class!

This Program has - Learning Rate scheduler with paitence value as 2. Feel free to change this as per requirement.
'''

# Import relevant Libraries

import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


'''
Train_model function takes in Data loader, model, optimizer, Criterion(loss function) and the device

'''
def train_model(data_loader, model, optimizer, criterion, device):
    loss_arr = []
    # Setting model to train mode
    model.train()
    
    for data in tqdm(data_loader, leave=True, desc='Training'):
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
        
        del inputs, target, output  # Deleteing data which will be not used again to free up GPU
    return round(sum(loss_arr)/len(loss_arr), 3)

'''
Evaluate_model takes in data_laoder, model and device
Tqdm has been used to show progress bar
'''
def evaluate_model(data_loader, model, device):
    
    # Set the model to evaluation mode
    model.eval()
    
    _actual = []
    _preds = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, leave=True, desc='Evaluating'):   # Tqdm - shows progress bar
            inputs = data["image"]
            target = data["target"]
            
            inputs = inputs.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            
            pred = model(inputs)
            
            target = target.detach().cpu().numpy().tolist()
            pred = pred.detach().cpu().numpy().tolist()
            
            _actual.extend(target)
            _preds.extend(pred)
    
    return metrics.roc_auc_score(_actual, _preds)   # It is returning ROC AUC Score. Change this as per requirement


'''
Select_model - Helps in selected models when we are trying out different architectures. Add more. 
Just pass - model name sauch as resnet18 or ResNet50 with pretrained.
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
If you want to configure the model - add or remove layers, you can define a function to make it look clean and easy!
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

# --------------- Now here we use all the functions defined --------------------
images = train_labels.img_path.values
targets = train_labels.target.values

aug = None
batch_size = 64

train_imgs, val_imgs, train_targets, val_targets = train_test_split(images, targets, 
                                                                    stratify=targets,
                                                                    random_state=42)

train_dataset = ET_Dataset(img_paths=train_imgs,
                                     targets=train_targets,
                                     resize=(224, 224),
                                     augmentations=aug)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=4)

val_dataset = ET_Dataset(img_paths=val_imgs,
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

model = get_configured_model_resnext(model_name = "resnext50", pretrained=True)
# Configure input layer as input has 6 channels and not 3
model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

epoch_loss = []
for epoch in range(epochs):
    mean_loss_batch = train_model(train_loader, model, optimizer, loss_fn, device)
    scheduler.step(mean_loss_batch)
    epoch_loss.append(mean_loss_batch)
    print("Epoch : {0}, Current Mean Loss : {1}".format(epoch+1, mean_loss_batch))
roc_val = evaluate_model(train_loader, model, device)
print("ROC AUC value is : ", roc_val)

plot_training_details(epoch_loss, lr, batch_size, epochs)