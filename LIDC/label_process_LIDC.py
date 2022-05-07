import glob
import os
import numpy as np
from PIL import Image

# merge all segments of four experts to a label image
# the lesion positions are those where at least one of the experts segmented a lesion.

# pic path
data_path = '../data/LIDC_DLCV_version'
sets = ['test']

for set in sets:
    set_path = os.path.join(data_path, set)
    image_paths = glob.glob(set_path + '/images/*.png')
    for image_path in image_paths:
        image = Image.open(image_path)
        # init a same size pic with all zero
        label = np.zeros(image.size)
        # read segments of each expert
        for i in range(4):
            path = image_path[:-4] + f'_l{i}.png'
            path = path.replace('images', 'lesions')
            img = Image.open(path)
            # add the segments together
            label += np.array(img).astype(np.float32)
        # if value is bigger than 0, the corresponding location is belongs to lesion
        label = np.where(label > 0, 255, 0)
        label_img = Image.fromarray(label).convert('L')
        label_img.save(image_path.replace('images', 'labels'))
