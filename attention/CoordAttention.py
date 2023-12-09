import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    # 使用 Relu 函数实现 Sigmoid 激活函数
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    # 使用 Relu 实现 Swish 激活函数
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        # 高度方向上的平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 宽度方向上的平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # 隐藏层
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # shortcut
        identity = x

        n, c, h, w = x.size()
        # nxcxhx1
        x_h = self.pool_h(x)
        # nxcx1xw -> nxcxwx1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 将 x_h, x_w 分开
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 注意力机制
        out = identity * a_w * a_h

        return out
