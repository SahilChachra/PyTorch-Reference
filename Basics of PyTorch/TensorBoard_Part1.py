'''
This py file demonstrates Tensorboard - demonstrates use of several batch_sizes, learning_rates

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter # Send data to Tensorboard

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8,
                               kernel_size=(3,3), stride=(1, 1), padding=(1, 1))
                                # Same Convolution. Input Output size remains same

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2)) # Half Dimension

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3),
                               stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking Train Acc')
    else:
        print('Checking Test Accuracy')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
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
        for x,y in loader:
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


model = CNN()
x = torch.rand(64, 1, 28, 28)
# print(x.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Loading Data
train_dataset = datasets.MNIST(root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network
model = CNN()
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Tensorboard Writer
# writer = SummaryWriter(f"runs/MNIST/trying_tensorboard")

# Which batchsize and Learning Rate go well together
batch_sizes = [1, 32, 64]
learning_rates = [0.1, 0.01, 0.001]


# Train Network
print('Model Training Started...')
print('Learning Rate = {0}, Optimizer = {1}, Loss Func = {2}, Device = {3}'.format(learning_rate, str(optimizer).split()[0],
                                                                                criterion, device))
loss_epoch_arr = []
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        model = CNN()
        model.to(device=device)
        model.train()
        writer = SummaryWriter(f"runs/MNIST/trying_tensorboard {batch_size} LR {learning_rate}")
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            losses= []
            accuracies = []
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

                losses.append(loss.item())

                # Calculate Running Loss
                _, preds = scores.max(1)
                num_correct = (preds == targets).sum()
                running_train_acc = float(num_correct)/float(data.shape[0]) #shape 0 gives batch. So per batch accuracy
                accuracies.append(running_train_acc)

                writer.add_scalar('Training Loss', loss, global_step=step)
                writer.add_scalar('Train Accuracy', running_train_acc, global_step=step)
                step +=1

            writer.add_hparams(
                {"lr": learning_rate, "bsize": batch_size},
                {
                    "accuracy": sum(accuracies) / len(accuracies),
                    "loss": sum(losses) / len(losses),
                },
            )
            loss_epoch_arr.append(loss.item())
            print('Epoch {0}/{1}. Train Accuracy : {2}. Loss = {3}'.format(epoch+1, num_epochs,
                                                                           accuracy_training(train_loader, model),
                                                                           round(loss.item(), 4)))


print('\nModel Training Completed. Average Loss = {0:.2f}'.format(sum(loss_epoch_arr)/len(loss_epoch_arr)))

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

if device=='cuda':
    torch.cuda.empty_cache()
exit()