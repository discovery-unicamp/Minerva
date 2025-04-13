from typing import Optional, Tuple
import torch
from minerva.models.nets.base import SimpleSupervisedModel


class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, activation_cls: torch.nn.Module):
        super().__init__()
        self.in_channels = in_channels
        self.activation_cls = activation_cls

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels=64, kernel_size=5, stride=1),
            torch.nn.BatchNorm1d(64),
            activation_cls(),
            torch.nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.block(x)


class SqueezeAndExcitation1D(torch.nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.num_channels_reduced = in_channels // reduction_ratio

        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_channels, self.num_channels_reduced),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_channels_reduced, in_channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        squeeze_tensor = input_tensor.mean(dim=2)
        x = self.block(squeeze_tensor)
        output_tensor = torch.mul(
            input_tensor,
            x.view(input_tensor.shape[0], input_tensor.shape[1], 1),
        )
        return output_tensor


class ResNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        activation_cls: torch.nn.Module = torch.nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.activation_cls = activation_cls

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            torch.nn.BatchNorm1d(32),
            activation_cls(),
            torch.nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            torch.nn.BatchNorm1d(64),
        )

    def forward(self, x):
        input_tensor = x
        x = self.block(x)
        x += input_tensor
        x = self.activation_cls()(x)
        return x


class ResNetSEBlock(ResNetBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block.append(SqueezeAndExcitation1D(64))


class _ResNet1D(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        residual_block_cls: type = ResNetBlock,
        activation_cls: type = torch.nn.ReLU,
        num_residual_blocks: int = 5,
        reduction_ratio: int = 2,
        avg_pooling: bool = True,
        **residual_block_cls_kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_residual_blocks = num_residual_blocks
        self.reduction_ratio = reduction_ratio

        self.conv_block = ConvolutionalBlock(
            in_channels=input_shape[0],
            activation_cls=activation_cls,
        )
        self.residual_blocks = torch.nn.Sequential(
            *[
                residual_block_cls(
                    in_channels=64,
                    activation_cls=activation_cls,
                    **residual_block_cls_kwargs,
                )
                for _ in range(num_residual_blocks)
            ]
        )
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.avg_pooling = avg_pooling

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        if self.avg_pooling:
            x = self.global_avg_pool(x)
            x = x.squeeze(2)
        return x


class ResNet1DBase(SimpleSupervisedModel):
    def __init__(
        self,
        resnet_block_cls: type = ResNetBlock,
        activation_cls: type = torch.nn.ReLU,
        input_shape: Tuple[int, int] = (6, 60),
        num_classes: int = 6,
        num_residual_blocks: int = 5,
        reduction_ratio=2,
        learning_rate: float = 1e-3,
        residual_block_cls_kwargs: Optional[dict] = None,
        # Arguments passed to the SimpleSupervisedModel constructor
        *args,
        **kwargs,
    ):
        backbone = _ResNet1D(
            input_shape=input_shape,
            residual_block_cls=resnet_block_cls,
            activation_cls=activation_cls,
            num_residual_blocks=num_residual_blocks,
            reduction_ratio=reduction_ratio,
            **(residual_block_cls_kwargs or {}),
        )

        self.fc_input_features = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = torch.nn.Linear(self.fc_input_features, num_classes)

        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            *args,
            **kwargs,
        )

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int, int]
            The input shape of the network.

        Returns
        -------
        int
            The number of features after the convolutional layers.
        """
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)


class ResNet1D_5(ResNet1DBase):
    def __init__(self, *args, **kwargs):
        if (
            "num_residual_blocks" in kwargs
            or "activation_cls" in kwargs
            or "resnet_block_cls" in kwargs
        ):
            raise ValueError(
                "`num_residual_blocks`, `activation_cls`, and `resnet_block_cls` "
                " should not be passed as arguments. Use ResNet1DBase instead."
            )

        super().__init__(
            *args,
            **kwargs,
            resnet_block_cls=ResNetBlock,
            activation_cls=torch.nn.ReLU,
            num_residual_blocks=5,
        )


# Deep Residual Network for Smartwatch-Based User Identification through Complex Hand Movements (ResNet1D)
class ResNet1D_8(ResNet1DBase):
    def __init__(self, *args, **kwargs):
        if (
            "num_residual_blocks" in kwargs
            or "activation_cls" in kwargs
            or "resnet_block_cls" in kwargs
        ):
            raise ValueError(
                "`num_residual_blocks`, `activation_cls`, and `resnet_block_cls` "
                " should not be passed as arguments. Use ResNet1DBase instead."
            )

        super().__init__(
            *args,
            **kwargs,
            resnet_block_cls=ResNetBlock,
            activation_cls=torch.nn.ELU,
            num_residual_blocks=8,
        )


# Deep Residual Network for Smartwatch-Based User Identification through Complex Hand Movements (ResNetSE1D)
class ResNetSE1D_8(ResNet1DBase):
    def __init__(self, *args, **kwargs):
        if (
            "num_residual_blocks" in kwargs
            or "activation_cls" in kwargs
            or "resnet_block_cls" in kwargs
        ):
            raise ValueError(
                "`num_residual_blocks`, `activation_cls`, and `resnet_block_cls` "
                " should not be passed as arguments. Use ResNet1DBase instead."
            )

        super().__init__(
            *args,
            **kwargs,
            resnet_block_cls=ResNetSEBlock,
            activation_cls=torch.nn.ELU,
            num_residual_blocks=8,
        )


# resnet-se: Channel Attention-Based Deep Residual Network for Complex Activity Recognition Using Wrist-Worn Wearable Sensors
# Changes the activation function to ReLU and the number of residual blocks to 5 (compared to ResNetSE1D_8)
class ResNetSE1D_5(ResNet1DBase):
    def __init__(self, *args, **kwargs):
        if (
            "num_residual_blocks" in kwargs
            or "activation_cls" in kwargs
            or "resnet_block_cls" in kwargs
        ):
            raise ValueError(
                "`num_residual_blocks`, `activation_cls`, and `resnet_block_cls` "
                " should not be passed as arguments. Use ResNet1DBase instead."
            )

        super().__init__(
            *args,
            **kwargs,
            resnet_block_cls=ResNetSEBlock,
            activation_cls=torch.nn.ReLU,
            num_residual_blocks=5,
        )
