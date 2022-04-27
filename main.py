import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from cityscape.city_dataset import CityDataset
from prob import ProbabilisticUnet
from utils import l2_regularisation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = CityDataset(location=r'D:\demo\\523\\final_pro\data\City\afterPre\half\train')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices), len(test_indices)))

net = ProbabilisticUnet(input_channels=3, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=2, no_convs_fcomb=4,
                        beta=10.0)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 10
for epoch in range(epochs):
    for step, (patch, mask) in enumerate(train_loader):
        patch = patch.to(device)
        mask = mask.to(device)
        # mask = torch.unsqueeze(mask, 1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
