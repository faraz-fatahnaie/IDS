import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, classification_mode: str = 'multi'):
        super(CNN, self).__init__()
        if classification_mode == 'multi':
            self.n_classes = 5
            self.activation = nn.Softmax(dim=1)
        elif classification_mode == 'binary':
            self.n_classes = 1
            self.activation = nn.Sigmoid()
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
        y = self.conv2d_1(x)
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
        y = nn.Flatten()(y)
        y = self.linear(y)
        # y = nn.Linear(in_features=y.size(1), out_features=self.n_classes).to('cuda:0')(y)
        y = self.activation(y)

        return y


class CNN_MAGNETO(nn.Module):
    def __init__(self, classification_mode: str = 'multi'):
        super(CNN_MAGNETO, self).__init__()
        if classification_mode == 'multi':
            self.n_classes = 5
            self.activation = nn.Softmax(dim=1)
        elif classification_mode == 'binary':
            self.n_classes = 1
            self.activation = nn.Sigmoid()

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1)

        self.linear0 = nn.Linear(in_features=15488, out_features=256)
        self.linear1 = nn.Linear(in_features=256, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=self.n_classes)

    def forward(self, x):
        y = self.conv0(x)
        y = F.relu(y)
        y = nn.Dropout()(y)

        y = self.conv1(y)
        y = F.relu(y)
        y = nn.Dropout()(y)

        y = self.conv2(y)
        y = F.relu(y)

        y = nn.Flatten()(y)
        y = self.linear0(y)
        y = F.relu(y)
        y = self.linear1(y)
        y = F.relu(y)
        y = self.linear2(y)
        y = self.activation(y)

        return y


# if __name__ == '__main__':
#     # summary(model=SEForward(), input_size=(1, 11, 11))
#     input = torch.randn(1, 1, 11, 11)
#     se = CNN()
#     output = se(input)
#     print(output.shape)
