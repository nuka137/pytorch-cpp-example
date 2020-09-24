import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        width = out_channels // 4

        self.conv1 = torch.nn.Conv2d(in_channels, width, kernel_size=(1, 1),
                                     stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(width)
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(width, width, kernel_size=(3, 3),
                                     stride=stride, padding=1, groups=1,
                                     bias=False, dilation=1)
        self.bn2 = torch.nn.BatchNorm2d(width)
        self.relu2 = torch.nn.ReLU(inplace=True)

        self.conv3 = torch.nn.Conv2d(width, out_channels, kernel_size=(1, 1),
                                     stride=1, padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)

        def shortcut(in_, out):
            if in_ != out:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(in_, out, kernel_size=(1, 1),
                                    stride=stride, padding=0, bias=False),
                    torch.nn.BatchNorm2d(out),
                )
            else:
                return lambda x: x
        self.shortcut = shortcut(in_channels, out_channels)

        self.relu3 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.shortcut(x)

        out = self.relu3(out + shortcut)

        return out


class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7),
                                     stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2,
                                          padding=1)

        self.layer1 = torch.nn.Sequential(
            ResidualBlock(64, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )

        self.layer2 = torch.nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )

        self.layer3 = torch.nn.Sequential(
            ResidualBlock(512, 1024, stride=2),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),
        )

        self.layer4 = torch.nn.Sequential(
            ResidualBlock(1024, 2048, stride=2),
            ResidualBlock(2048, 2048),
            ResidualBlock(2048, 2048),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten(1)
        self.fc = torch.nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

