"""Training loop.  Usage: python -m src.train --data_dir data/processed --epochs 10"""
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import HistopathologyDataset, TRAIN_TRANSFORM, DEFAULT_TRANSFORM
from src.model import build_model


def train(data_dir, epochs, batch_size, lr, backbone, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full = HistopathologyDataset(data_dir)
    if len(full) == 0:
        print("No data found. Run scripts/sort_breakhis.py first.")
        return

    # proper train/val split (separate transforms, no shared-object bug)
    samples = full.samples[:]
    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(0.2 * len(samples)))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    train_set = HistopathologyDataset(data_dir, TRAIN_TRANSFORM, samples=train_samples)
    val_set = HistopathologyDataset(data_dir, DEFAULT_TRANSFORM, samples=val_samples)
    print(f"train={len(train_set)}  val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = build_model(backbone=backbone, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_set)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_loss += criterion(out, labels).item() * images.size(0)
                correct += (out.argmax(1) == labels).sum().item()
        val_loss /= len(val_set)
        val_acc = correct / len(val_set)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)
            print(f"  -> saved best model to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    p.add_argument("--out_path", default="models/best_model.pt")
    a = p.parse_args()
    train(a.data_dir, a.epochs, a.batch_size, a.lr, a.backbone, a.out_path)
