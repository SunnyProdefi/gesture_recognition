import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class GestureRecognizer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # 去除最后分类层
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x).view(B, T, -1)  # shape: [B, T, 512]
        pooled = feats.mean(dim=1)  # shape: [B, 512]
        return self.fc(pooled)
