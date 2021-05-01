import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import sys

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking Train Acc')
    else:
        print('Checking Test Accuracy')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)  # 64x10 -> 64 images, 10 outputs
            _, predictions = scores.max(1)  # max from 2nd dimension. Just need index and not value
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            del x, y
        print("Accuracy : {0:.2f}".format((num_correct / num_samples).item() * 100))

    model.train()


def accuracy_training(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)  # 64x10 -> 64 images, 10 outputs
            _, predictions = scores.max(1)  # max from 2nd dimension. Just need index and not value
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            # print("Accuracy : {0:.2f}".format((num_correct / num_samples).item() * 100))
            del x, y

    model.train()
    return round((num_correct / num_samples).item() * 100, 2)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyper parameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5

# Loading Data
train_dataset = datasets.CIFAR10(root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# test_dataset = datasets.CIFAR10(root='./dataset/', train=False, transform=transforms.ToTensor(), download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# Load model
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model = torchvision.models.vgg16(pretrained=True)
print(model)

for param in model.parameters():
    param.requires_grad = False

# model.avgpool = Identity()
model.classifier[6] = nn.Linear(4096, 10)
print(model)

# Initialize Network
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
print('Model Training Started...')
print('Learning Rate = {0}, Optimizer = {1}, Loss Func = {2}, Device = {3}'.format(learning_rate,
                                                                                   str(optimizer).split()[0],
                                                                                   criterion, device))
loss_epoch_arr = []
min_loss = 999
for epoch in range(num_epochs):

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del data, targets, scores

    loss_epoch_arr.append(loss.item())
    print('Epoch {0}/{1}. Train Accuracy : {2}. Loss = {3}'.format(epoch + 1, num_epochs,
                                                                   accuracy_training(train_loader, model),
                                                                   round(loss.item(), 4)))

print('\nModel Training Completed. Average Loss = {0:.2f}'.format(sum(loss_epoch_arr) / len(loss_epoch_arr)))

check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)

if device == 'cuda':
    torch.cuda.empty_cache()
exit()
