import torch
import os
from torch.utils.data import Dataset
from skimage import io
from CustomDatasetFileName import ClassName
import torchvision.transforms as transforms
from torchvision.utils import save_image

#Transforms

my_transforms = transforms.Compose([    # Compose concatenates serveral tranforms. It does in that in order given.
    tranforms.ToPILImage(), # All transforms work on PIL Image
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.5), # Randomly change brightness
    transforms.RandomRotation(degree=45),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) #Calculate mean value for all pixels across
    # all channels and for std also. (pixel_value - mean)/ std <- for each channel

])

dataset = ClassName(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized',
                    transform = my_tranforms)

# Save image to visualize
for i in range(10):
    for img, label in dataset:
        save_image(img, 'Label of image')