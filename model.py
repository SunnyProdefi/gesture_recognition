# 导入 PyTorch 中的神经网络模块
import torch.nn as nn
# 导入 torchvision 中的 ResNet18 模型和其预训练权重
from torchvision.models import resnet18, ResNet18_Weights

# 定义手势识别网络，继承自 nn.Module
class GestureRecognizer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()  # 调用父类的构造函数
        # 加载带预训练权重的 ResNet18 模型（用于提取图像特征）
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # 去掉 ResNet18 的最后一个全连接分类层，只保留前面的特征提取层（包含卷积、残差块、池化等）
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # 输出特征为 [B, 512, 1, 1]
        # 添加一个新的线性层，将 512 维特征映射为类别数，用于最终分类
        self.fc = nn.Linear(512, num_classes)

    # 定义前向传播逻辑
    def forward(self, x):
        # 输入张量 x 的维度为 [B, T, C, H, W]，表示：
        # B: 批量大小（视频数）
        # T: 每个视频的帧数
        # C: 图像通道数（RGB 为 3）
        # H, W: 图像高和宽
        B, T, C, H, W = x.size()  # 解包张量维度信息
        # 将视频数据展开成 T 张图片的批量，维度变为 [B*T, C, H, W]，以便送入 CNN 提取特征
        x = x.view(B * T, C, H, W)
        # 使用 ResNet18 提取每帧图像的特征，输出为 [B*T, 512, 1, 1]，再 reshape 成 [B, T, 512]
        feats = self.feature_extractor(x).view(B, T, -1)
        # 对时间维度进行平均池化（取 T 帧的平均特征），输出为 [B, 512]
        pooled = feats.mean(dim=1)
        # 使用全连接层将 512 维特征映射为 num_classes 维的类别概率分布
        return self.fc(pooled)
