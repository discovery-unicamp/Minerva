import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x = torch.add(x, identity)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, layer_sizes, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        intermidiate_channels = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]

        # Essentially the entire ResNet architecture are in these 5 lines below
        self.layers = nn.ModuleList([])

        for i in range(len(layer_sizes)):
            layer = self.make_layer(layer_sizes[i], intermidiate_channels[i], strides[i])
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(intermidiate_channels[-1] * 4, num_classes)

    def make_layer(self, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        # The expansion size is always 4 for ResNet 50,101,152
        expansion = intermediate_channels * 4

        if stride != 1 or self.in_channels != expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(expansion),
            )

        layers.append(
            ResNetBlock(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        self.in_channels = expansion

        for _ in range(num_residual_blocks - 1):
            layers.append(ResNetBlock(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet([3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet([3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet([3, 8, 36, 3], img_channel, num_classes)
