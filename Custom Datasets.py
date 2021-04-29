import torch
import os
from torch.utils.data import Dataset
from skimage import io

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

# How to use the Dataset

from CustomDatasetFileName import ClassName

dataset = ClassName(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized',
                    transform = transform.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [2000, 5000])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)