import os
import json
import sys

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm

from my_dataset import SurfaceDefectDataset
from my_dataset import defect_labels
from model import SurfaceDectectResNet


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'using {device}')

    # 创建训练集
    train_dataset = SurfaceDefectDataset(r'C:\Users\Administrator\Desktop\data\deep_learning\enu_surface_defect\train')
    train_num = len(train_dataset)

    # 类别和index的对应关系, 写入文件.
    cla_dict = dict((i, label) for i, label in enumerate(defect_labels))
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'using {nw} dataloader workers every process')

    # 把dataset变成dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = SurfaceDefectDataset(r'C:\Users\Administrator\Desktop\data\deep_learning\enu_surface_defect\test')
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    net = SurfaceDectectResNet(num_classes=6)
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    epochs = 10
    save_path = './model.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    # 训练过程
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data['image'], data['defect']
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f'train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}'

        # 校验代码
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data['image'], val_data['defect']
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accuracy = acc / val_num
        print(f'[epoch {epoch + 1} train_loss: {running_loss / train_steps:.3f},'
              f'val_accuracy:{val_accuracy:.3f}')
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()