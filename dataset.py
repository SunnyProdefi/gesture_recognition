import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class GestureVideoDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, frames_per_video=30):
        self.data = pd.read_csv(csv_path, sep=';', header=None, names=["video_id", "gesture", "label"])
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_folder = os.path.join(self.root_dir, row.iloc[0])
        label = int(row.iloc[2])

        # 获取所有图片文件，按文件名排序
        img_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        img_files.sort()
        # 取前frames_per_video张图片
        selected_imgs = img_files[:self.frames_per_video]

        frames = []
        for img_name in selected_imgs:
            img_path = os.path.join(video_folder, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        # 如果图片数量不足，补齐最后一帧
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1].clone())

        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, label
