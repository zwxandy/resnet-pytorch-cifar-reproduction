import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniONN(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniONN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(16, num_classes)
        self.maxpool = nn.MaxPool2d()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # MiniONN from COINN: 6Conv + 1FC + 2MP + 6ReLU
        
        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))
        x = self.maxpool(x)

        x = self.relu(self.conv5(x))

        x = self.relu(self.conv6(x))

        x = self.linear(self.flatten(x))
        x = F.softmax(x, dim=-1)

        return x 