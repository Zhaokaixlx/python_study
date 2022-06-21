import os
import json

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2 as cv

from model import SurfaceDectectResNet


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                    transforms.Resize((200, 200))
        ])
    img_path = "Sc_215.bmp"
    assert os.path.exists(img_path), f'{img_path} does not exist'

    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), f'{json_path} does not exist'
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    model = SurfaceDectectResNet(num_classes=6).to(device)
    weights_path = './model.pth'
    assert os.path.exists(weights_path), f'{weights_path} does not exist'
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = model(img.to(device))
        output = torch.squeeze(output).cpu()
        predict = torch.softmax(output, dim=0)
        predict_class = torch.argmax(predict).numpy()

    print_res = f'class: {class_dict[str(predict_class)]}, prob:{predict[predict_class].numpy()}'
    plt.title(print_res)
    plt.show()


if __name__ == '__main__':
    main()