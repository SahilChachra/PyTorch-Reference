import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}  # Flatten the layers


class VGG16(nn.Module):
    def __init__(self, in_channles=3, num_classes=1000, vgg_arch=VGG_types["VGG16"]):
        super(VGG16, self).__init__()
        self.in_channels = in_channles
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(vgg_arch)

        self.fcs = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channles = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channles, out_channels=out_channels, kernel_size=(3, 3),
                                     stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]

                in_channles = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


vgg16 = VGG16(in_channles=3, num_classes=1000,vgg_arch=VGG_types["VGG16"])
x = torch.rand(1, 3, 224, 224)
print(vgg16(x).shape)

print('Number of Parameters :', sum(p.data.nelement() for p in vgg16.parameters()))
print(vgg16.train())