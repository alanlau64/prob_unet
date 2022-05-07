import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from cityscape.city_dataset import CityDataset
from prob import ProbabilisticUnet
from unet import Unet
from utils import l2_regularisation
import gc


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


def train(batch_size, epochs, gpu, val_after=100, lr=1e-4):
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu) else 'cpu')
    train_dataset = CityDataset(location='../data/postprocess/city/quarter/train')
    val_dataset = CityDataset(location='../data/postprocess/city/quarter/val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Number of training/validation patches:", (len(train_dataset), len(val_dataset)))

    net = ProbabilisticUnet(input_channels=3, num_classes=34)
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
                    torch.save(net.state_dict(), "./probmodel_city")
                print("-------------------")
    np.save('loss_train_city', training_losses)
    np.save('loss_val_city', val_losses)


def test(gpu):
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu) else 'cpu')
    test_dataset = CityDataset(location='../data/postprocess/city/quarter/test')
    test_loader = DataLoader(test_dataset, batch_size=1)
    net = ProbabilisticUnet(input_channels=3, num_classes=34)
    net.load_state_dict(torch.load("./probmodel_city"))
    net.to(device)
    iou = 0
    cnt = 0
    for step, (patch, mask) in enumerate(test_loader):
        with torch.no_grad():
            patch = patch.to(device)
            mask = mask.to(device)
            net.forward(patch, None, training=False)
            out = net.sample(testing=True)
            print(out.squeeze()[0])
            # out = net.sample(testing
            iou += mIOU(mask, out, 34)
            cnt += 1

    print(f"Mean IoU = {iou / cnt * 100}%")
