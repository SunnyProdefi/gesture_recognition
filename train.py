import torch # 导入 PyTorch 主库
from torch.utils.data import DataLoader  # 用于加载数据集
from torchvision import transforms  # 图像预处理模块
from dataset import GestureVideoDataset  # 自定义数据集类
from model import GestureRecognizer  # 手势识别模型
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块

# === 路径配置（根据你本地的数据路径修改）===
train_csv = "data/archive/train.csv"  # 训练集标签 CSV 文件路径
train_root = "data/archive/train"     # 训练集视频帧文件夹路径
val_csv = "data/archive/val.csv"      # 验证集标签 CSV 文件路径
val_root = "data/archive/val"         # 验证集视频帧文件夹路径

# === 定义图像预处理方法 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像尺寸统一为 224x224（适配 ResNet）
    transforms.ToTensor(),          # 将 PIL 图像转换为 PyTorch 张量，自动归一化到 [0,1]
])

print("Loading datasets...")

# === 加载训练集与验证集 ===
train_set = GestureVideoDataset(train_csv, train_root, transform)
val_set = GestureVideoDataset(val_csv, val_root, transform)
# 使用 DataLoader 封装数据集，设置批大小为 4，训练时打乱顺序
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

print("Datasets loaded.")
# 输出数据统计信息
print(f"Total training samples: {len(train_set)}")  # 训练样本数量
print(f"Batch size: {train_loader.batch_size}")     # 批大小
print(f"Each epoch will process {len(train_loader)} batches.")  # 每轮训练批次数

# === 初始化模型与优化器 ===
model = GestureRecognizer()  # 实例化模型
# 设置设备为 GPU（若可用）或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到指定设备
print(f"Using device: {device}")
print("Model initialized.")

# 定义损失函数：交叉熵损失用于分类任务
criterion = nn.CrossEntropyLoss()
# 定义优化器：Adam，学习率设为 0.0005
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print("Starting training...")

# === 主训练循环（共训练 10 个 epoch）===
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    model.train()  # 切换到训练模式（启用 dropout、BN 等）
    print("Training...")
    total_loss = 0  # 累加每轮的总损失

    # 遍历每个 batch
    for batch_idx, (videos, labels) in enumerate(train_loader, start=1):
        print(f"[Epoch {epoch+1}] Batch {batch_idx}/{len(train_loader)}")

        # 将视频数据和标签发送到设备（GPU 或 CPU）
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)  # 前向传播，得到预测结果
        loss = criterion(outputs, labels)  # 计算损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()        # 反向传播
        optimizer.step()       # 参数更新
        total_loss += loss.item()  # 累加 batch 损失

    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")  # 打印每轮训练损失

    # === 验证阶段 ===
    model.eval()  # 切换为评估模式（禁用 dropout、BN）
    correct = 0   # 正确分类的样本数
    total = 0     # 总样本数
    with torch.no_grad():  # 禁用梯度计算，加速推理
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 获取每个样本预测概率最大的类别索引
            total += labels.size(0)  # 累加样本数
            correct += (predicted == labels).sum().item()  # 累加预测正确的样本数

    # 输出验证集准确率
    print(f"Validation Accuracy: {correct / total:.2%}")
