import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class SEBlock2D(nn.Module):
    def __init__(self, in_channels: int, ratio=16):
        super(SEBlock2D, self).__init__()
        self.se_shape = (-1, in_channels, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_features=in_channels, out_features=in_channels // ratio, bias=False)
        self.linear2 = nn.Linear(in_features=in_channels // ratio, out_features=in_channels, bias=False)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        in_data = x
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = x.view(self.se_shape)
        x = torch.multiply(in_data, x)
        return x


class SE(nn.Module):
    def __init__(self, classification_mode):
        super(SE, self).__init__()
        if classification_mode == 'multi':
            self.n_classes = 5
            self.activation = nn.Softmax(dim=1)
        elif classification_mode == 'binary':
            self.n_classes = 1
            self.activation = nn.Sigmoid()

        self.SE = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=1),
            nn.ReLU(True),
            SEBlock2D(in_channels=8, ratio=4),
            # nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            # SEBlock2D(in_channels=16, ratio=4),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Dropout(0.1)

        )
        self.fc = nn.Linear(in_features=1352, out_features=self.n_classes)

    def forward(self, x):
        out = self.SE(x)
        out = self.fc(out)

        return self.activation(out)


if __name__ == '__main__':
    # summary(model=SE(classification_mode='binary'), input_size=(1, 14, 14))
    se = SE(classification_mode='binary')
    output = se(torch.randn(1, 1, 14, 14))
    # print(output.shape)
