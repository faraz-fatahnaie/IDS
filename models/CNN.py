import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.n_classes = 5
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.linear = nn.Linear(in_features=64, out_features=self.n_classes)
        self.bn_1 = nn.BatchNorm2d(16)
        self.bn_2 = nn.BatchNorm2d(32)
        self.bn_3 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = x
        y = self.conv2d_1(y)
        y = self.bn_1(y)
        y = F.relu(y)
        y = self.maxpool2d_1(y)

        y = self.conv2d_2(y)
        y = self.bn_2(y)
        y = F.relu(y)
        y = self.maxpool2d_2(y)

        y = self.conv2d_3(y)
        y = self.bn_3(y)
        y = F.relu(y)

        y = self.gap(y)
        #print(y.size()[1])
        y = nn.Flatten()(y)
        #print(y.size()[1])
        y = self.linear(y)
        y = F.softmax(y, dim=1)

        return y


if __name__ == '__main__':
    #summary(model=SEForward(), input_size=(1, 11, 11))
    input = torch.randn(1, 1, 11, 11)
    se = CNN()
    output = se(input)
    print(output.shape)