# train_v2.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1) Involution layer & ASDNet (Version 2) ---

class Involution2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 reduction_ratio: int = 4):
        super(Involution2d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio
        self.o = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, kernel_size * kernel_size, 1)
        )
        self.unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.o(x)
        b, c, h, w = x.shape
        x_unfold = self.unfold(x).view(b, self.in_channels, self.kernel_size*self.kernel_size, h, w)
        kernel = kernel.view(b, 1, self.kernel_size*self.kernel_size, h, w)
        out = (kernel * x_unfold).sum(dim=2)
        return out

class ASDNet(nn.Module):
    def __init__(self, input_channels=3):
        super(ASDNet, self).__init__()
        self.involution_block = nn.Sequential(
            Involution2d(input_channels, kernel_size=3, stride=1, reduction_ratio=2),
            Involution2d(input_channels, kernel_size=3, stride=1, reduction_ratio=2),
            Involution2d(input_channels, kernel_size=3, stride=1, reduction_ratio=2),
            nn.ReLU()
        )
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.dense_block = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.involution_block(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        return self.dense_block(x)

# --- 2) Data transforms & local paths ---

data_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Update these to wherever you extracted your ZIPs:
dataset1_path = r"./dataset1/Images"
dataset2_path = r"./dataset2/Images"

dataset1 = datasets.ImageFolder(dataset1_path, transform=data_transform)
dataset2 = datasets.ImageFolder(dataset2_path, transform=data_transform)

# Augment by simple concatenation (4×)
augmented_dataset1 = ConcatDataset([dataset1] * 4)
augmented_dataset2 = ConcatDataset([dataset2] * 4)
merged_augmented_dataset = ConcatDataset([augmented_dataset1, augmented_dataset2])

print(f"Dataset1 size (augmented): {len(augmented_dataset1)}")
print(f"Dataset2 size (augmented): {len(augmented_dataset2)}")
print(f"Merged size (augmented):    {len(merged_augmented_dataset)}")

def get_loaders(dataset, batch_size=32):
    n = len(dataset)
    train_n = int(0.8 * n)
    val_n   = int(0.1 * n)
    test_n  = n - train_n - val_n
    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    )

train_loader_d1, val_loader_d1, test_loader_d1 = get_loaders(augmented_dataset1)
train_loader_d2, val_loader_d2, test_loader_d2 = get_loaders(augmented_dataset2)
train_loader_merged, val_loader_merged, test_loader_merged = get_loaders(merged_augmented_dataset)

# --- 3) Training & evaluation, with torch.save() ---

def train_and_evaluate(train_loader, val_loader, test_loader, dataset_name, epochs=45):
    print(f"\n=== Training on {dataset_name} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASDNet(input_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels)
        val_acc = corrects.double() / len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs}  Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                f"asdnet_v2_best_{dataset_name}.pth"
            )

    # Testing
    print(f"\n--- Testing on {dataset_name} ---")
    model.load_state_dict(torch.load(f"asdnet_v2_best_{dataset_name}.pth", map_location=device))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc   = accuracy_score(all_labels, all_preds)
    recall= recall_score(all_labels, all_preds, average='macro')
    f1    = f1_score(all_labels, all_preds, average='macro')
    print(f"Test Results for {dataset_name}: Acc={acc*100:.2f}%, Recall={recall*100:.2f}%, F1={f1*100:.2f}%")

if __name__ == "__main__":
    train_and_evaluate(train_loader_d1, val_loader_d1, test_loader_d1, "Dataset1")
    train_and_evaluate(train_loader_d2, val_loader_d2, test_loader_d2, "Dataset2")

