import torch
import torch.nn as nn
from typing import Any, Literal, Optional
from minerva.models.nets.base import SimpleSupervisedModel


class ResNetBlock(nn.Module):
    """
    Implementation of a single ResNet block.
    """

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        identity_downsample: Optional[torch.nn.Module] = None,
        stride: int = 1
    ):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels to the block.
        intermediate_channels : int
            The number of channels in the intermediate convolutional layers within the block.
        identity_downsample : nn.Module, optional
            A downsampling layer to match the dimensions of the input and output if they differ. 
            If `None`, no downsampling is performed. Default is `None`.
        stride : int, optional
            The stride value for the first convolutional layer in the block. It determines the 
            downsampling factor for the spatial dimensions. Default is `1`.
        """

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


class _ResNet(torch.nn.Module):
    """Implementation of ResNet model."""

    def __init__(
            self,
            layer_sizes: list[int],
            image_channels: int,
            num_classes: int,
        ):
        """Implementation of ResNet model.

        Parameters
        ----------
        layer_sizes : list of int
            A list specifying the number of layers in each residual block stage. For example:
            - ResNet-50: [3, 4, 6, 3]
            - ResNet-101: [3, 4, 23, 3]
            - ResNet-152: [3, 8, 36, 3]
        image_channels : int
            The number of channels in the input image, typically 3 for RGB images or 1 for grayscale.
        num_classes : int
            The number of output classes for the classification task.
        """

        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        intermidiate_channels = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]

        self.layers = nn.ModuleList([])

        for i in range(len(layer_sizes)):
            layer = self.make_layer(layer_sizes[i], intermidiate_channels[i], strides[i])
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(intermidiate_channels[-1] * 4, num_classes)

    def make_layer(self, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
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

class ResNet(SimpleSupervisedModel):
    """
    This class is a simple implementation of the ResNet (Residual Network) model, 
    which is widely used in image classification and other computer vision tasks. 
    The ResNet architecture introduces residual connections, allowing deeper networks 
    to be trained by mitigating the vanishing gradient problem. The model consists 
    of repeated building blocks with skip connections that add the input of a 
    layer to its output after applying transformations. ResNet was originally 
    proposed by He et al. in 2015.

    This implementation supports ResNet-50, ResNet-101, and ResNet-152, offering 
    flexibility in network depth based on the specific use case. The model can 
    handle arbitrary input sizes and supports both RGB and grayscale images.

    References
    ----------
    He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings 
    of the IEEE conference on computer vision and pattern recognition. 2016.

    

    Notes
    -----
    - The expected input size is (N, C, H, W), where:
      - N is the batch size,
      - C is the number of channels,
      - H is the height of the input image, and
      - W is the width of the input image.
    - The output shape is (N, num_classes), where `num_classes` corresponds to 
      the number of classes specified during initialization.
    """

    def __init__(
        self,
        type: Literal["50", "101", "152"] = "50",
        img_channel: int = 3, 
        num_classes: int = 1000,
        learning_rate: float = 1e-3,
        loss_fn: Optional[torch.nn.Module] = None,
        **kwargs: dict[str, Any],
    ):
        """Wrapper implementation of the ResNet model.

        Parameters
        ----------
        type : Literal["50", "101", "152"], optional
            The type of ResNet architecture to use. Options are:
            - "50": ResNet-50
            - "101": ResNet-101
            - "152": ResNet-152
            Default is "50".
        img_channel : int, optional
            The number of channels in the input image, by default 3 (for RGB images).
        num_classes : int, optional
            The number of output classes for the classification task, by default 1000.
        learning_rate : float, optional
            The learning rate for the Adam optimizer, by default 1e-3.
        loss_fn : torch.nn.Module, optional
            The function used to compute the loss. If `None`, `MSELoss` will be used, 
            by default None.
        kwargs : dict
            Additional arguments to be passed to the `SimpleSupervisedModel` class.
        """
        resnet_type = { "50": [3, 4, 23, 3], "101": [3, 4, 23, 3], "152": [3, 8, 36, 3] } 
        backbone = _ResNet(layer_sizes=resnet_type[type], image_channels=img_channel, num_classes=num_classes)

        super().__init__(
            backbone=backbone,
            fc=torch.nn.Identity(),
            loss_fn=loss_fn or torch.nn.CrossEntropyLoss(),
            learning_rate=learning_rate,
            flatten=False,
            **kwargs,
        )