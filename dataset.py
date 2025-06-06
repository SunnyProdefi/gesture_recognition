# 导入所需的库
import os  # 用于处理文件和目录路径
import torch  # PyTorch 主库
from torch.utils.data import Dataset  # 用于自定义数据集类
from PIL import Image  # 用于图像读取
import pandas as pd  # 用于读取 CSV 文件

# 自定义手势视频数据集类，继承自 PyTorch 的 Dataset 基类
class GestureVideoDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, frames_per_video=30):
        # 读取 CSV 文件，使用分号分隔，无表头，列名设为 video_id, gesture（手势名）, label（标签索引）
        self.data = pd.read_csv(csv_path, sep=';', header=None, names=["video_id", "gesture", "label"])
        # 指定图像根目录（即所有视频帧图像所在的根文件夹）
        self.root_dir = root_dir
        # 图像变换操作（如 resize, normalize, ToTensor 等），默认为 None
        self.transform = transform
        # 每个视频最多加载的帧数，默认是 30 帧
        self.frames_per_video = frames_per_video

    # 返回数据集中视频的数量（即 CSV 文件中的行数）
    def __len__(self):
        return len(self.data)

    # 获取指定索引 idx 对应的视频帧和标签
    def __getitem__(self, idx):
        # 获取第 idx 行数据
        row = self.data.iloc[idx]
        # 构建当前视频帧所在的文件夹路径：root_dir/video_id
        video_folder = os.path.join(self.root_dir, row.iloc[0])
        # 获取标签值，并转为整数类型
        label = int(row.iloc[2])

        # 获取该视频文件夹下所有图像文件（以 .jpg/.jpeg/.png 结尾，不区分大小写）
        img_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # 将图像文件名按字典序排序，确保帧顺序一致
        img_files.sort()
        # 取前 frames_per_video 帧（最多 frames_per_video 个）
        selected_imgs = img_files[:self.frames_per_video]

        frames = []  # 存储所有帧图像的列表
        for img_name in selected_imgs:
            # 构建图像文件完整路径
            img_path = os.path.join(video_folder, img_name)
            # 读取图像并转换为 RGB 格式（确保通道一致）
            image = Image.open(img_path).convert('RGB')
            # 如果定义了图像变换操作，则应用变换
            if self.transform:
                image = self.transform(image)
            # 将图像添加到帧列表中
            frames.append(image)

        # 如果图像帧数量不足 frames_per_video，则复制最后一帧补齐
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1].clone())  # 使用 clone 确保生成新的张量副本

        # 将所有帧图像堆叠成一个 tensor，形状为 [帧数, 通道数, 高度, 宽度]
        video_tensor = torch.stack(frames, dim=0)
        # 返回视频帧序列和对应的标签
        return video_tensor, label
