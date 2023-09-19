import torch
from torch import nn


class SPConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, alpha: float = 0.5):
        super(SPConv2D, self).__init__()
        assert 0 <= alpha <= 1
        self.alpha = alpha

        self.in_rep_channels = int(in_channels * self.alpha)
        self.out_rep_channels = int(out_channels * self.alpha)
        self.out_channels = out_channels
        self.stride = stride

        self.represent_gp_conv = nn.Conv2d(in_channels=self.in_rep_channels,
                                           out_channels=self.out_channels,
                                           stride=self.stride,
                                           kernel_size=3,
                                           padding=1,
                                           groups=2)
        self.represent_pt_conv = nn.Conv2d(in_channels=self.in_rep_channels,
                                           out_channels=out_channels,
                                           kernel_size=1)

        self.redundant_pt_conv = nn.Conv2d(in_channels=in_channels - self.in_rep_channels,
                                           out_channels=out_channels,
                                           kernel_size=1)

        self.avg_pool_s2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.avg_pool_add_1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_add_3 = nn.AdaptiveAvgPool2d(1)

        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.group = int(1 / self.alpha)

    def forward(self, x):
        batch_size = x.size()[0]

        x_3x3 = x[:, :self.in_rep_channels, ...]
        x_1x1 = x[:, self.in_rep_channels:, ...]
        rep_gp = self.represent_gp_conv(x_3x3)

        if self.stride == 2:
            x_3x3 = self.avg_pool_s2_3(x_3x3)
        rep_pt = self.represent_pt_conv(x_3x3)
        rep_fuse = rep_gp + rep_pt
        rep_fuse = self.bn1(rep_fuse)
        rep_fuse_ration = self.avg_pool_add_3(rep_fuse).squeeze(dim=3).squeeze(dim=2)

        if self.stride == 2:
            x_1x1 = self.avg_pool_s2_1(x_1x1)

        red_pt = self.redundant_pt_conv(x_1x1)
        red_pt = self.bn2(red_pt)
        red_pt_ratio = self.avg_pool_add_1(red_pt).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((rep_fuse_ration, red_pt_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)

        out_mul_1 = red_pt * (out_31_ratio[:, :, 1].view(batch_size, self.out_channels, 1, 1).expand_as(red_pt))
        out_mul_3 = rep_fuse * (out_31_ratio[:, :, 0].view(batch_size, self.out_channels, 1, 1).expand_as(rep_fuse))

        return out_mul_1 + out_mul_3


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


class SPSE(nn.Module):
    def __init__(self):
        super(SPSE, self).__init__()
        self.se = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=1),
            nn.ReLU(True),
            # nn.Dropout(),
            SEBlock2D(in_channels=8, ratio=4),
            nn.ReLU(True),
            nn.Flatten(),
            # nn.Dropout()

        )

        self.sp = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=1),
            nn.ReLU(True),
            # nn.Dropout(),
            SPConv2D(in_channels=8, out_channels=16, stride=1, alpha=0.8),
            nn.ReLU(True),
            # SPConv2D(in_channels=16, out_channels=32, stride=1, alpha=0.8),
            # nn.ReLU(True),
            nn.Flatten(),
            # nn.Dropout()
        )

        self.fc = nn.Linear(in_features=4056, out_features=1)

    def forward(self, x):
        out_se = self.se(x)
        out_sp = self.sp(x)

        out = torch.concatenate([out_se, out_sp], dim=1)
        out = self.fc(out)
        return nn.Sigmoid()(out)


if __name__ == '__main__':
    # from torchsummary import summary

    # model = SPConv2D(in_channels=248, out_channels=512, alpha=0.6)
    # summary(model.cuda(), (248, 100, 100))
    #
    # model2 = SPConv1D(in_channels=248, out_channels=512, alpha=0.6)
    # summary(model2, [248, 1000])

    model = SPSE()
    model(torch.randn((1, 1, 14, 14)))
