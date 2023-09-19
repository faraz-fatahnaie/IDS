import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


class ECA(nn.Module):
    def __init__(self):
        super(ECA, self).__init__()
        self.n_classes = 1
        self.eca = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=1),
            # nn.ReLU(True),
            nn.Sigmoid(),
            # ECAAttention(kernel_size=3),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            ECAAttention(kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            # nn.ReLU(True),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(in_features=1024, out_features=self.n_classes),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.eca(x)


if __name__ == '__main__':
    input = torch.randn(1, 1, 14, 14)
    eca = ECAAttention(kernel_size=3)
    output = eca(input)
    print(output.shape)
