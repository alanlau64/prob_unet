import glob
import os
import numpy as np
from PIL import Image

data_path = '../data/LIDC_DLCV_version'
sets = ['test', 'train', 'val']

for set in sets:
    set_path = os.path.join(data_path, set)
    image_paths = glob.glob(set_path + '/images/*.png')
    for image_path in image_paths:
        image = Image.open(image_path)
        label = np.zeros(image.size)
        for i in range(4):
            path = image_path[:-4] + f'_l{i}.png'
            path = path.replace('images', 'lesions')
            img = Image.open(path)
            label += np.array(img).astype(np.float32)
        label = np.where(label > 0, 255, 0)
        label_img = Image.fromarray(label).convert('L')
        label_img.save(image_path.replace('images', 'labels'))
