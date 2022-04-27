import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from LIDC.load_LIDC import LIDC
# from cityscape.city_dataset import CityDataset
import torch.nn.functional as F
from prob import ProbabilisticUnet
from utils import l2_regularisation


def mIOU(label, pred, num_classes=19):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


torch.random.manual_seed(42)
device = torch.device('cuda' if False else 'cpu')
# train_dataset = CityDataset(location='../data/postprocess/city/quarter/train')
# val_dataset = CityDataset(location='../data/postprocess/city/quarter/val')
# test_dataset = CityDataset(location='../data/postprocess/city/quarter/test')
train_dataset = LIDC('train', transform=torchvision.transforms.ToTensor())
test_dataset = LIDC('test', transform=torchvision.transforms.ToTensor())
val_dataset = LIDC('val', transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=10)
val_loader = DataLoader(val_dataset, batch_size=10)
test_loader = DataLoader(test_dataset, batch_size=1)
print("Number of training/val patches/test patches:", (len(train_dataset), len(val_loader), len(test_loader)))

net = ProbabilisticUnet(input_channels=1, num_classes=1)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 0
current_step = 0
max_val_loss = 100000
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    for step, (patch, mask) in enumerate(train_loader):
        patch = patch.to(device)
        mask = mask.to(device)
        current_step += 1
        if current_step % 25 == 0:
            print(f"Step {step + 1} of {len(train_loader)}")
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        if current_step % 25 == 0:
            print(f"Training Loss = {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if current_step % 100 == 0:
            print("Start validating")
            print("------------------")
            net.eval()
            val_loss = 0
            for step, (val_patch, val_mask) in enumerate(val_loader):
                val_patch = val_patch.to(device)
                val_mask = val_mask.to(device)
                net.forward(val_patch, val_mask, training=False)
                val_elbo = net.elbo(val_mask, analytic_kl=False)
                val_loss += -elbo + 1e-5 * l2_regularisation(net.posterior) + l2_regularisation(
                    net.prior) + l2_regularisation(net.fcomb.layers)
            val_loss /= len(val_loader)
            print(f"Validation Loss = {val_loss}")
            print("-------------------")
            if val_loss < max_val_loss:
                print("Saving model")
                max_val_loss = val_loss
                torch.save(net.state_dict(), "./unetmodel")

net.load_state_dict(torch.load("./unetmodel"))
iou = 0
cnt = 0
acc = 0
print("Start testing")
print("------------------")
for step, (patch, mask) in enumerate(test_loader):
    with torch.no_grad():
        patch = patch.to(device)
        mask = mask.to(device)
        net.forward(patch, None, training=False)
        out = net.sample(testing=True)
        if out.shape[1] == 1:
            out = torch.cat((out, 1 - out), dim=1)
        iou += mIOU(mask, out, 34)
        out = torch.max(out, 1, True).indices
        acc += (torch.sum(out == mask) / mask.nelement())
        cnt += 1

print(f"Accuracy = {acc / cnt * 100}%")
print(f"Mean IoU = {iou / cnt * 100}%")
