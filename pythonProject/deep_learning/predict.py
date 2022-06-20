import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_path = './4.jpeg'
    assert os.path.exists(img_path), f'{img_path} does not exist'
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    # print(img.shape)
    img = torch.unsqueeze(img, dim=0)
    # print(img.shape)
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f'file {json_path} does not exist'
    with open(json_path, 'r') as f:
        class_dict = json.load(f)
    # print(class_dict)

    model = AlexNet(num_classes=5).to(device)
    weights_path = './AlexNet.pth'
    assert os.path.exists(weights_path), f'file {weights_path} does not exist'
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = model(img.to(device))
        # print(output)
        output = torch.squeeze(output).cpu()
        predict = torch.softmax(output, dim=0)
        predict_class = torch.argmax(predict).numpy()

    print_res = f"class: {class_dict[str(predict_class)]}, prob: {predict[predict_class].numpy():.3f}"
    plt.title(print_res)
    plt.show()

if __name__ == '__main__':
    main()