import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle

import glob
import PIL.Image as Image


# class LIDC_IDRI(Dataset):
#     images = []
#     labels = []
#     series_uid = []

#     def __init__(self, dataset_location, transform=None):
#         self.transform = transform
#         max_bytes = 2 ** 31 - 1
#         data = {}
#         for file in os.listdir(dataset_location):
#             filename = os.fsdecode(file)
#             if '.pickle' in filename:
#                 print("Loading file", filename)
#                 file_path = dataset_location + filename
#                 bytes_in = bytearray(0)
#                 input_size = os.path.getsize(file_path)
#                 with open(file_path, 'rb') as f_in:
#                     for _ in range(0, input_size, max_bytes):
#                         bytes_in += f_in.read(max_bytes)
#                 new_data = pickle.loads(bytes_in)
#                 data.update(new_data)

#         for key, value in data.items():
#             self.images.append(value['image'].astype(float))
#             self.labels.append(value['masks'])
#             self.series_uid.append(value['series_uid'])

#         assert (len(self.images) == len(self.labels) == len(self.series_uid))

#         for img in self.images:
#             assert np.max(img) <= 1 and np.min(img) >= 0
#         for label in self.labels:
#             assert np.max(label) <= 1 and np.min(label) >= 0

#         del new_data
#         del data

#     def __getitem__(self, index):
#         image = np.expand_dims(self.images[index], axis=0)

#         # Randomly select one of the four labels for this image
#         label = self.labels[index][random.randint(0, 3)].astype(float)
#         if self.transform is not None:
#             image = self.transform(image)

#         series_uid = self.series_uid[index]

#         # Convert image and label to torch tensors
#         image = torch.from_numpy(image)
#         label = torch.from_numpy(label)

#         # Convert uint8 to float tensors
#         image = image.type(torch.FloatTensor)
#         label = label.type(torch.FloatTensor)

#         return image, label, series_uid

#     # Override to give PyTorch size of dataset
#     def __len__(self):
#         return len(self.images)

class LIDC(torch.utils.data.Dataset):
    def __init__(self, set, transform, data_path='./data/LIDC_DLCV_version'):
        'Initialization'
        self.transform = transform

        data_path = os.path.join(data_path, set)
        self.image_paths = glob.glob(data_path + '/images/*.png')  # changes jpg to png

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = np.zeros(image.size)
        for i in range(4):
            path = image_path[:-4] + f'_l{i}.png'
            path = path.replace('images', 'lesions')
            img = Image.open(path)
            label += np.array(img).astype(np.float32)
        label = np.where(label>0, 1, 0)

        # add convert to jpg
        y = self.transform(label)
        X = self.transform(image)
        return X, y

    def get_num_classes(self):
        return 2


if __name__ == '__main__':
    dataset = LIDC(train=True, transform=None, data_path='./LIDC_crops/LIDC_DLCV_version')
    print(dataset.name_to_label)
