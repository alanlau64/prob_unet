import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from prob import ProbabilisticUnet
device = torch.device('cuda' if False else 'cpu')
net = ProbabilisticUnet(input_channels=1, num_classes=1)
net.to(device)
net.load_state_dict(torch.load("./unetmodel"))
img_path = './data/LIDC_DLCV_version/train/images/LIDC-IDRI-0001_z-120.0_c0.png'
test_img = Image.open(img_path)
transform = torchvision.transforms.ToTensor()
test_img = transform(test_img).unsqueeze(0)
test_img.to(device)
net.forward(test_img, None, training=False)
out = net.sample(testing=True)
if out.shape[1] == 1:
    out = torch.cat((out, 1 - out), dim=1)
out = torch.max(out, 1, True).indices
out = out.squeeze(0)
out = out.squeeze(0).numpy()
img = Image.fromarray(out).convert('L')
plt.imshow(img)
plt.show()