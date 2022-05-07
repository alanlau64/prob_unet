import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from unet import Unet
from LIDC.load_LIDC import LIDC
# from cityscape.city_dataset import CityDataset
import torch.nn.functional as F
from prob import ProbabilisticUnet
from utils import l2_regularisation


def iou_1class(outputs: torch.Tensor, labels: torch.Tensor):
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    SMOOTH = 1e-6
    labels = labels / labels.max()
    outputs = outputs / outputs.max()
    outidx = (outputs != 0)
    labelidx = (labels != 0)
    intersection = outidx[labelidx].long().sum()
    union = outidx.long().sum() + (labelidx.long().sum()) - intersection
    iou = (float(intersection) + SMOOTH) / (float(union) + SMOOTH)
    return iou  # Or thresholded.mean() if you are interested in average across the batch


torch.random.manual_seed(42)

def train(batch_size, epochs, gpu, val_after=100, lr=1e-4):
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu) else 'cpu')
    train_dataset = LIDC('train', transform=torchvision.transforms.ToTensor())
    val_dataset = LIDC('val', transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Number of training/validation patches:", (len(train_dataset), len(val_dataset)))

    net = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=2)
    # net = Unet(3, 34, [32, 64, 128, 192], {'w': 'he_normal', 'b': 'normal'},
    #           apply_last_layer=True, padding=True)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    max_val_loss = 100000
    current_step = 0
    training_losses = []
    val_losses = []
    total_step = len(train_loader) * epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for step, (patch, mask) in enumerate(train_loader):
            current_step += 1
            if current_step % 25 == 0:
                print(f"Step {current_step} of {total_step}")
            net.train()
            patch = patch.to(device)
            mask = mask.to(device)
            # out = net.forward(patch, False)
            # criterion = torch.nn.CrossEntropyLoss()
            # loss = criterion(out, mask.long().squeeze(1))
            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            if current_step % 25 == 0:
                print(f"Training Loss = {loss.item()}")
            if current_step % val_after == 0:
                print(f"Recording Loss")
                training_losses.append(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if current_step % val_after == 0:
                print("------------------")
                print("Start validating")
                net.eval()
                val_loss = 0
                with torch.no_grad():
                    for step, (val_patch, val_mask) in enumerate(val_loader):
                        val_patch = val_patch.to(device)
                        val_mask = val_mask.to(device)
                        # val_out = net.forward(val_patch, False)
                        # val_criterion = torch.nn.CrossEntropyLoss()
                        # val_loss += val_criterion(val_out,
                        #                           val_mask.long().squeeze(1))
                        net.forward(val_patch, None)
                        val_elbo = net.elbo(val_mask, analytic_kl=False)
                        val_loss += -val_elbo + 1e-5 * reg_loss
                    val_loss = val_loss / len(val_loader)
                    print(f"Validation Loss = {val_loss}")
                    val_losses.append(val_loss.detach())
                if val_loss < max_val_loss:
                    print("Saving model")
                    max_val_loss = val_loss.detach()
                    torch.save(net.state_dict(), "./probmodel_lidc")
                print("-------------------")
    np.save('loss_train_lidc', training_losses)
    np.save('loss_val_lidc', val_losses)


def test(gpu):
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu) else 'cpu')
    test_dataset = LIDC('test', transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1)
    net = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=2)
    net.load_state_dict(torch.load("./probmodel_lidc"))
    net.to(device)
    iou = 0
    empty = 0
    cnt = 0
    print("Start testing")
    print("------------------")
    for step, (patch, mask) in enumerate(test_loader):
        with torch.no_grad():
            patch = patch.to(device)
            mask = mask.to(device)
            net.forward(patch, None, training=False)
            out = net.sample(testing=True)
            # out = net.forward(patch, False)
            if out.shape[1] == 1:
                out = torch.cat((out, 1 - out), dim=1)
            out = torch.max(out, 1, True).indices
            iou += iou_1class(out, mask)
            if mask.sum() == 0:
                empty += 1
            cnt += 1

    print(f"Mean IoU = {iou / (cnt - empty) * 100}%")