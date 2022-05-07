import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from prob import ProbabilisticUnet

# method to produce a segment of the original pic
device = torch.device('cuda' if False else 'cpu')
net = ProbabilisticUnet(input_channels=1, num_classes=1)
net.to(device)
# load the trained model
net.load_state_dict(torch.load("./unetmodel", map_location=device))
img_path = './data/LIDC_DLCV_version/test/images/LIDC-IDRI-0014_z-135.25_c0.png'
# read the pic
test_img = Image.open(img_path)
transform = torchvision.transforms.ToTensor()
test_img = transform(test_img).unsqueeze(0)
test_img.to(device)
# produce the segment
net.forward(test_img, None, training=False)
out = net.sample(testing=True)
# if the channel of the output is 1, it is a single classification problem. Then concatenate the output with 1 - output
if out.shape[1] == 1:
    out = torch.cat((out, 1 - out), dim=1)
# pick the index of the bigger one
out = torch.max(out, 1, True).indices
out = out.squeeze(0)
# highlight the lesion location with 255 in the grey pic
out = np.uint8(out.squeeze(0).numpy() * 255)
img = Image.fromarray(out).convert('L')
plt.imshow(img)
plt.show()
img.save('./data/LIDC_DLCV_version/result/LIDC-IDRI-0014_z-135.25_c0.png')