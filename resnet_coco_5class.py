# ResNet Image Classifier (COCO 5-Class Subset)
#
# script trains a small ResNet using skip connections
# on a subset from the COCO dataset:
# person
#  chair
#  car
#    dining table
#    bottle



import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from pycocotools.coco import COCO
import torchvision.transforms as T


# Decide whether to use CPU or GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COCO paths
# using val2017 for both train & val
COCO_ROOT = Path("coco")
TRAIN_IMAGES = COCO_ROOT / "val2017"    
VAL_IMAGES = COCO_ROOT / "val2017"
ANN_FILE = COCO_ROOT / "annotations" / "instances_val2017.json"

# The 5 target classes chosen (appear frequently in COCO)
CLASSES = ["person", "chair", "car", "dining table", "bottle"]
NUM_CLASSES = len(CLASSES)


IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 3           # short training due to  my bad CPU 
LR = 1e-3

# Keep dataset size rsmall so training finishes within minutes
MAX_SAMPLES_PER_CLASS = 2000

# Output folders
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")



# DATASET: COCO bounding-box crop loader


class COCOSubset(Dataset):
    """
    Custom dataset for cropping COCO bounding boxes and treating
    them as single-object classification samples.

    This converts a detection dataset into a classification dataset.
    """

    def __init__(self, ann_file, img_dir, class_names, transform=None):
        super().__init__()
        self.coco = COCO(str(ann_file))
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform

        # Map each class name to the matching COCO category ID
        self.class_to_catid = {
            c: self.coco.getCatIds(catNms=[c])[0] for c in class_names
        }

        # Convert class names → numerical labels
        self.class_to_label = {c: i for i, c in enumerate(class_names)}

        self.samples = []             # list of (filename, bbox, label)
        self.class_counts = np.zeros(len(class_names), dtype=int)

        print("\n=== Building COCO subset dataset ===")

        # Loop through each class and extract its bounding boxes
        for cname in class_names:
            cat_id = self.class_to_catid[cname]
            label = self.class_to_label[cname]

            img_ids = self.coco.getImgIds(catIds=[cat_id])
            per_class_counter = 0

            for img_id in img_ids:
                if per_class_counter >= MAX_SAMPLES_PER_CLASS:
                    break  # stop collecting once class limit reached

                ann_ids = self.coco.getAnnIds(
                    imgIds=[img_id], catIds=[cat_id], iscrowd=False
                )
                anns = self.coco.loadAnns(ann_ids)
                img_info = self.coco.loadImgs([img_id])[0]
                file_name = img_info["file_name"]

                for ann in anns:
                    if per_class_counter >= MAX_SAMPLES_PER_CLASS:
                        break

                    bbox = ann["bbox"]
                    self.samples.append((file_name, bbox, label))
                    self.class_counts[label] += 1
                    per_class_counter += 1

            print(f"{cname:12s}: {self.class_counts[label]} samples (after cap)")

        print(f"Total samples in dataset: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, bbox, label = self.samples[idx]
        img_path = self.img_dir / file_name

        # Load & crop the bounding box region
        img = Image.open(img_path).convert("RGB")
        x, y, w, h = bbox
        img = img.crop((int(x), int(y), int(x + w), int(y + h)))

        # Apply transforms (resize, augment, normalize)
        if self.transform:
            img = self.transform(img)

        return img, label


#001364457 CHEEMA
# MODEL: Small ResNet (skip connections included)


class ResidualBlock(nn.Module):
    """
    Basic Residual Block:
    Conv → BN → ReLU → Conv → BN → +skip → ReLU
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut path fixes dimension mismatches
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # skip connection
        return F.relu(out)


class SmallResNet(nn.Module):
    """
    Lightweight ResNet-style architecture.
    Balanced for CPU-speed and decent performance.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.in_ch = 32

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 3 stacked residual “stages”
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        layers = [ResidualBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(blocks - 1):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



def create_weighted_sampler(dataset: COCOSubset):
    """
    Oversample minority classes based on inverse frequency.
    Helps fix imbalance in COCO where 'person' dominates heavily.
    """
    labels = np.array([lbl for _, _, lbl in dataset.samples])

    counts = np.bincount(labels, minlength=NUM_CLASSES)
    print("\nClass counts (after cap):", counts)

    class_weights = 1.0 / (counts + 1e-6)
    print("Class weights:", class_weights)

    # Assign a weight to each sample
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights),
                                 len(sample_weights),
                                 replacement=True)


def train_one_epoch(model, loader, optimizer, criterion, epoch_idx, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        # Progress print every few batches
        if batch_i % 50 == 0:
            print(f"  [Epoch {epoch_idx}/{total_epochs}] Batch {batch_i}/{len(loader)}")

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    """
    Evaluate model on validation set.
    No gradient updates here.
    """
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def plot_curves(train_acc, val_acc, train_loss, val_loss):
    """
    Save accuracy and loss curves for the report/presentation.
    """
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy curve
    plt.figure()
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(plots_dir / "acc_curve.png")
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()



# MAIN TRAINING LOOP


def main():
    print("Using device:", DEVICE)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Training transforms add small variations for oaquiaracy
    transform_train = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(),
        T.ToTensor(),
    ])


    transform_val = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])

    print("\nLoading COCO subset for TRAIN...")
    train_ds = COCOSubset(ANN_FILE, TRAIN_IMAGES, CLASSES, transform_train)

    print("\nLoading COCO subset for VAL...")
    val_ds = COCOSubset(ANN_FILE, VAL_IMAGES, CLASSES, transform_val)

    sampler = create_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build model & optimizer
    model = SmallResNet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    print("\n=== Starting training ===")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, EPOCHS
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.3f} | "
              f"Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.3f}")

        # Save best performing model
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_resnet_coco_5cls.pth")

    print("\nBest val acc:", best_val_acc)
    plot_curves(train_accs, val_accs, train_losses, val_losses)
    print("Training finished. Plots saved in results/plots/")


if __name__ == "__main__":
    main()
#001364457 CHEEMA