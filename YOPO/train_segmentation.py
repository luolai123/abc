import os
import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from segmentation.dataset import RGBObstacleDataset, list_dataset_statistics
from segmentation.model import SegmentationUNet


def get_dataloaders(data_root: str, image_size: Tuple[int, int], batch_size: int):
    train_ds = RGBObstacleDataset(data_root, image_size=image_size, mode="train")
    val_ds = RGBObstacleDataset(data_root, image_size=image_size, mode="valid")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Valid", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary segmentation network for obstacle masks")
    parser.add_argument("--data_root", required=True, help="Dataset root, containing rgb/ and mask/ folders")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--log_dir", default="./saved/segmentation")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    print("==== Dataset Summary ====")
    for line in list_dataset_statistics(args.data_root):
        print(line)

    image_size = (args.height, args.width)
    train_loader, val_loader = get_dataloaders(args.data_root, image_size, args.batch_size)

    model = SegmentationUNet()
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    writer = SummaryWriter(log_dir=args.log_dir)

    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = evaluate(model, val_loader, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.log_dir, f"segmentation_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

