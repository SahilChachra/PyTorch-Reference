'''
This is a Training Pipeline implementing Ensembling method of the classifier. For this we will also use StratifiedKFold split to have different models trained on diff parts of the dataset

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

# --------------- Now here we use all the functions defined --------------------

# Configure model
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.BCEWithLogitsLoss()

X = train_data_csv.img_path.values
Y = train_data_csv.target.values
skf = StratifiedKFold(n_splits=5)
fold = 0
lr = 5e-4
batch_size = 32
epochs = 5

for train_index, test_index in skf.split(X, Y):
    
    model = get_configured_model_resnet("resnet50", true)
    model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Edit input layer as requirement

    model.to(device)

    train_images, valid_images = X[train_index], X[test_index]
    train_targets, valid_targets = Y[train_index], Y[test_index]

    train_dataset = YourDataSet(image_paths=train_images, targets=train_targets)
    valid_dataset = YourDataSet(image_paths=valid_images, targets=valid_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_loss = []
    for epoch in range(epochs):
        mean_loss_batch = train_model(train_loader, model, optimizer, loss_fn, device)
        epoch_loss.append(mean_loss_batch)
        print("Epoch : {0}, Current Mean Loss : {1}".format(epoch+1, mean_loss_batch))

    _actual, _preds = evaluate_model(train_loader, model, device)  # You can evaluate the model after each epoch also!
    roc_val = metrics.roc_auc_score(_actual, _preds)
    print("ROC AUC value is : ", roc_val)

    # Save each model after each epoch
    torch.save(model.state_dict() + '-ResNet-' + str(fold) + '.pt')
    models.append(model)
    fold += 1

plot_training_details(epoch_loss, lr, batch_size, epochs)

# Create a TestLoader and submissions is a file which stores the path of input images and targets

test_dataset = YourDataSet(image_paths=submission.img_path.values, targets=submission.target.values)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ++++++++++ Using Ensembling +++++++++

sig = torch.nn.Sigmoid()
outs = []
for model in models:
    predictions, valid_targets = evaluate_model(test_loader, model, device=device)
    predictions = np.array(predictions)[:, 0]
    out = sig(torch.from_numpy(predictions))
    out = out.detach().numpy()
    outs.append(out)