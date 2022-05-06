import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os

import glob
import PIL.Image as Image


# dataset of LIDC
class LIDC(torch.utils.data.Dataset):
    def __init__(self, set, transform, data_path='./data/LIDC_DLCV_version'):
        self.transform = transform
        # path of all original pic
        data_path = os.path.join(data_path, set)
        self.image_paths = glob.glob(data_path + '/images/*.png')  # changes jpg to png

    def __len__(self):
        # the length of the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        # get the path of label image
        label_path = self.image_paths[idx].replace('images', 'labels')
        label = Image.open(label_path)
        label = np.array(label).astype(np.float32)
        # if value is bigger than 0, the corresponding location is belongs to lesion
        label = np.where(label > 0, 1, 0)
        # transfer numpy to tensor
        y = self.transform(label)
        X = self.transform(image)
        return X, y

    def get_num_classes(self):
        return 2


if __name__ == '__main__':
    dataset = LIDC(train=True, transform=None, data_path='./LIDC_crops/LIDC_DLCV_version')
    print(dataset.name_to_label)
