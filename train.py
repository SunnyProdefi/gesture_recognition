import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import GestureVideoDataset
from model import GestureRecognizer
import torch.nn as nn
import torch.optim as optim

# 路径配置（你需要根据你本地调整）
train_csv = "data/archive/train.csv"
train_root = "data/archive/train"
val_csv = "data/archive/val.csv"
val_root = "data/archive/val"

# 数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("Loading datasets...")

# 数据加载
train_set = GestureVideoDataset(train_csv, train_root, transform)
val_set = GestureVideoDataset(val_csv, val_root, transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

print("Datasets loaded.")
# 输出统计信息
print(f"Total training samples: {len(train_set)}")
print(f"Batch size: {train_loader.batch_size}")
print(f"Each epoch will process {len(train_loader)} batches.")

# 模型与优化器
model = GestureRecognizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")  # device is already set to cuda/cpu
print("Model initialized.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print("Starting training...")
# 训练循环
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    model.train()
    print("Training...")
    total_loss = 0
    for batch_idx, (videos, labels) in enumerate(train_loader, start=1):
        print(f"[Epoch {epoch+1}] Batch {batch_idx}/{len(train_loader)}")

        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {correct / total:.2%}")

