"""Evaluation.  Usage: python -m src.evaluate --data_dir data/processed --weights models/best_model.pt"""
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.dataset import HistopathologyDataset, DEFAULT_TRANSFORM
from src.model import load_trained_model, CLASS_NAMES


def evaluate(data_dir, weights_path, backbone, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = HistopathologyDataset(data_dir, transform=DEFAULT_TRANSFORM)
    if len(dataset) == 0:
        print("No data found under", data_dir)
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = load_trained_model(weights_path, backbone=backbone, device=device)

    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(device))
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())

    acc = accuracy_score(labels_all, preds_all)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="binary")
    cm = confusion_matrix(labels_all, preds_all)
    print(f"Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {rec:.4f}\nF1-score:  {f1:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"           {CLASS_NAMES[0]:>10} {CLASS_NAMES[1]:>10}")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>10} {row[0]:>10} {row[1]:>10}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--weights", default="models/best_model.pt")
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()
    evaluate(a.data_dir, a.weights, a.backbone, a.batch_size)
