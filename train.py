import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.tensorboard import SummaryWriter

from dataset import FullDataset
from DCSUNet import DCSUNet
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("DCS-UNet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=50,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


writer = SummaryWriter()

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def main(args):
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    model = DCSUNet(args.hiera_path)
    model.to(device)
    optim = opt.AdamW([{"params": model.parameters(), "initia_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)

    # Early Stopping Parameters
    patience = 5  # 容忍 5 个 epoch 无显著提升
    early_stop_counter = 0  # 记录未改进的 epoch 数

    best_loss = float('inf')

    for epoch in range(args.epoch):
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()

            total_loss += loss.item()
            writer.add_scalar('Training Loss/Batch', loss.item(), epoch * len(dataloader) + i)

            if i % 50 == 0:
                print(f"epoch:{epoch + 1}-{i + 1}: loss:{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Training Loss/Epoch', avg_loss, epoch + 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0  # 重置计数器
            # torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
            # print(f"[Best Model Saved] Epoch {epoch + 1} with Loss: {best_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"[No Improvement] Early Stop Counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("[Early Stopping Triggered] Training stopped due to no improvement.")
            break  # 触发提前停止

        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'DCS-{epoch + 1}.pth'))
            print(f'[Saving Snapshot:] DCS-{epoch + 1}.pth')

        scheduler.step()

    writer.close()

if __name__ == "__main__":
    main(args)
