import os

import numpy as np
from torch.utils.data import Dataset


def Myloader(path):
    return np.load(path)


class CityDataset(Dataset):
    def __init__(self, location, transform=None):
        self.location = location
        self.transform = transform
        self.loader = Myloader
        self.image_path = []
        self.label_path = []
        self.loadPath()

    def __getitem__(self, item):
        image, label = self.image_path[item], self.label_path[item]
        image = self.loader(image)
        label = self.loader(label)
        return image, label

    def __len__(self):
        return len(self.image_path)

    def loadPath(self):
        city_names = os.listdir(self.location)
        for city in city_names:
            city_dir = os.path.join(self.location, city)
            image_names = os.listdir(city_dir)
            for image_name in image_names:
                if image_name.endswith('leftImg8bit.npy'):
                    self.image_path.append(os.path.join(self.location, city, image_name))
                    self.label_path.append(os.path.join(self.location, city, image_name[:-15]+'gtFine_labelIds.npy'))
                else:
                    continue
