import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
asdnet_v2_best_MergedDataset.pth
# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Involution layer & ASDNet (Version 2) ---

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

# --- Data transforms & paths ---

data_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset1_path = r"./dataset1/Images"
dataset2_path = r"./dataset2/Images"

dataset1 = datasets.ImageFolder(dataset1_path, transform=data_transform)
dataset2 = datasets.ImageFolder(dataset2_path, transform=data_transform)

augmented_dataset1 = ConcatDataset([dataset1] * 4)
augmented_dataset2 = ConcatDataset([dataset2] * 4)
merged_augmented_dataset = ConcatDataset([augmented_dataset1, augmented_dataset2])

# --- Data loaders ---
def get_loaders(dataset, batch_size=32):
    n = len(dataset)
    train_n = int(0.8 * n)
    val_n   = int(0.1 * n)
    test_n  = n - train_n - val_n
    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )

train_loader, val_loader, test_loader = get_loaders(merged_augmented_dataset)

# --- Training ---
def train_and_evaluate(train_loader, val_loader, test_loader, epochs=45):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASDNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        corrects = sum((model(inputs.to(device)).argmax(1) == labels.to(device)).sum().item() for inputs, labels in val_loader)
        val_acc = corrects / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "asdnet_v2_best_MergedDataset.pth")

    # Evaluate on test
    model.load_state_dict(torch.load("asdnet_v2_best_MergedDataset.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            preds = model(inputs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Acc={acc*100:.2f}%, Recall={recall*100:.2f}%, F1={f1*100:.2f}%")

if __name__ == '__main__':
    train_and_evaluate(train_loader, val_loader, test_loader)
