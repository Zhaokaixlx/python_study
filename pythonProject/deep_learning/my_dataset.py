# 创建自己的数据集, 集成自pytorch的dataset类, 必须实现__len__和__getitem__
import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv

# CR 裂纹 crackle
# In 夹杂 inclusion
# SC 划痕 scratch
# PS 压入氧化皮  press in oxide scale
# RS 麻点
# PA 斑点
defect_labels = ['In', 'Sc', 'Cr', 'PS', 'RS', 'Pa']


class SurfaceDefectDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.Resize((200, 200))
        ])
        img_files = os.listdir(root_dir)
        self.defect_types = []
        self.images = []
        for file_name in img_files:
            # 以下划线分割文件名
            defect_class = file_name.split('_')[0]
            defect_index = defect_labels.index(defect_class)
            self.images.append(os.path.join(root_dir, file_name))
            self.defect_types.append(defect_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        img = cv.imread(image_path) # BGR
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        sample = {'image': self.transform(img), 'defect': self.defect_types[idx]}
        return sample


if __name__ == '__main__':
    ds = SurfaceDefectDataset(r'C:\Users\Administrator\Desktop\data\deep_learning\enu_surface_defect\train')
    print(len(ds))
    print(ds[0]['image'].shape, ds[0]['defect'])
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8)
    sample = next(iter(dl))
    print(type(sample))
    print(sample['image'].shape)