import os
import json
import sys

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets

from model import AlexNet


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'using {device}')

    image_path = os.path.join('./', 'flower_data')
    # print(image_path)
    assert os.path.exists(image_path), "image path does not exist"

    data_transform = {
            'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'val': transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # 创建训练dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                         transform=data_transform['train'])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    print(flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'using {nw} dataloader workers every process')
    # 把dataset变成dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                batch_size=batch_size, shuffle=True,
                                num_workers=nw)

    # images, labels = next(iter(train_loader))
    # print(labels.numpy())
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                            transform=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=nw)

    net = AlexNet(num_classes=5)

    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_step = len(train_loader)

    # 训练过程
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f'train epoch {epoch + 1}/ {epochs} loss: {loss:.3f}'

        # 校验过程
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accuracy = acc / val_num
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_step:.3f}, val_accuracy:{val_accuracy:.3f}')
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()