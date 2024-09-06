import torch
import torch.nn as nn

class ResNet12(nn.Module):
    def __init__(self):
        super(ResNet12, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2)
        self.fc = nn.Linear(256 * 7 * 7, 10)

    def _make_layer(self, in_planes, out_planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU())
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(-1, 256 * 7 * 7)
        x = self.fc(x)
        return x