import torch
import torchvision


class SurfaceDectectResNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.cnn_layers = torchvision.models.resnet18(pretrained=True)
        in_features = self.cnn_layers.fc.in_features
        self.cnn_layers.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        out = self.cnn_layers(x)
        return out
