"""Dataset expecting data/processed/benign/ and data/processed/malignant/."""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


class HistopathologyDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, samples=None):
        self.transform = transform or DEFAULT_TRANSFORM
        self.class_to_idx = {"benign": 0, "malignant": 1}
        if samples is not None:
            self.samples = samples
            return
        self.samples = []
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    self.samples.append((os.path.join(class_dir, fname), label))
        if not self.samples:
            print(f"Warning: no images found under {root_dir} (need benign/ and malignant/ subfolders).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
