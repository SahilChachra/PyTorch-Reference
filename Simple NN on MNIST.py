# Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Connected Neural Network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NN(784, 10)
x = torch.rand(64, 784)
print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data
train_dataset = datasets.CIFAR10(root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# Initialize Network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        """print(data.shape)
        Here the input is in shape 64x1x28x28 where 64 is batch size
        We want image 28x28 to be 784"""

        data = data.reshape(data.shape[0], -1)
        # print(data.shape)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad() # Clear prev gradients
        loss.backward() # Perform back propagation
        optimizer.step() # Take a step of optimizer


# Check accuracy

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking Training Accuracy")
    else:
        print("Checking Test Accuracy")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # For checking, dont calculate gradients. no use
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x) # 64x10 -> 64 images, 10 outputs
            _, predictions = scores.max(1) #max from 2nd dimension. Just need index and not value
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)

        print("Accuracy : {0:.2f}".format((num_correct/num_samples).item()*100))

    model.train()  # Tells that we are training the model
    # model.eval() # Tells that we are testing the model
    # hence layers like BN will be fronzen else it'll throw error

check_accuracy(train_loader, model)
