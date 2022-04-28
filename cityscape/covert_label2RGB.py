import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cityscape.labels_config import labels

colors = []
for label in labels:
    colors.append(list(label[7]))

label_img = Image.open(r'D:\demo\523\City\gtFine\train\aachen\aachen_000000_000019_gtFine_labelIds.png')

label_array = np.array(label_img).astype(int)
color_array = np.array(colors)[label_array]
img = Image.fromarray(np.uint8(color_array)).convert('RGB')
plt.imshow(img)
plt.show()
