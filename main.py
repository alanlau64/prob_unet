import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from cityscape.city_dataset import CityDataset
from prob import ProbabilisticUnet
from unet import Unet
from utils import l2_regularisation

torch.random.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = CityDataset(location='./Testdata/postprocess/city/quarter/train')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=10, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices), len(test_indices)))

# net = ProbabilisticUnet(input_channels=3, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=2, no_convs_fcomb=4,
#                         beta=10.0)
net = Unet(3, 34, [32, 64, 128, 192], {'w': 'he_normal', 'b': 'normal'},
                         apply_last_layer=True, padding=True)
net.to(device)
# net.load_state_dict(torch.load("unetmodel"))

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    for step, (patch, mask) in enumerate(train_loader):
        print(f"Step {step + 1} of {len(train_loader)}")
        patch = patch.to(device)
        mask = mask.to(device)
        # # mask = torch.unsqueeze(mask, 1)
        # net.forward(patch, mask, training=True)
        # elbo = net.elbo(mask)
        # reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        # loss = -elbo + 1e-5 * reg_loss
        # print(loss.item())
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        out = net.forward(patch, False)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out, mask.long().squeeze(1))
        print(f"Loss = {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(net.state_dict(), "./unetmodel")

acc = 0
cnt = 0
for step, (patch, mask) in enumerate(test_loader):
    with torch.no_grad():
        out = net.forward(patch, True)
        out = torch.max(out, 1, True).indices
        acc += (torch.sum(out == mask) / mask.nelement())
        cnt += 1

print(f"Accuracy = {acc / cnt}")