import numpy as np
import torch
import torchvision
from PIL import Image

SMOOTH = 1e-6


def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs / outputs.max()
    labels = labels / labels.max()
    outidx = (outputs==1)
    labelidx = (labels!=0)
    intersection = outputs[labelidx].long().sum().items()
    union = outputs.long().sum().items() + (labels.long().sum().items()) - intersection
    iou = (float(intersection) + SMOOTH) / (float(union) + SMOOTH)
    print(iou)
    return iou  # Or thresholded.mean()


img_path = './data/LIDC_DLCV_version/result/LIDC-IDRI-0007_z-102.0_c0.png'
test_img = Image.open(img_path)
test_array = np.array(test_img).astype(np.uint8)
print(test_array.shape)
img_path = './data/LIDC_DLCV_version/test/labels/LIDC-IDRI-0007_z-102.0_c0.png'
real_img = Image.open(img_path)
real_array = np.array(real_img).astype(np.uint8)
print(real_array.shape)
loss = iou_numpy(test_array, real_array)
print(loss)
