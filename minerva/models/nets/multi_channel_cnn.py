from typing import Dict, Tuple
import torch
import lightning as L
from torchmetrics import Accuracy

from minerva.models.nets.base import SimpleSupervisedModel


class _MultiChannelCNN_HAR(torch.nn.Module):
    def __init__(self, input_channels: int = 1, concatenate: bool = True):
        super().__init__()

        self.freq_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels, 16, kernel_size=2, stride=1, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=2, stride=1, padding="same"),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.welch_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels, 16, kernel_size=2, stride=1, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=2, stride=1, padding="same"),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.concatenate = concatenate

    def forward(self, x):
        # Input is a 5D tensor (Batch, Transformed, Channel, H, W)
        # X[:, 0, :, :, :] --> Frequency (1, 6, 60)
        # X[:, 1, :, :, :] --> Welch     (1, 6, 60)
        freq_out = self.freq_encoder(x[:, 0, :, :, :])
        welch_out = self.welch_encoder(x[:, 1, :, :, :])
        if not self.concatenate:
            return freq_out, welch_out
        else:
            return torch.cat([freq_out, welch_out], dim=1)


class MultiChannelCNN_HAR(SimpleSupervisedModel):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        """Create a simple 1D Convolutional Network with 3 layers and 2 fully
        connected layers.

        Parameters
        ----------
        input_shape : Tuple[int, int], optional
            A 2-tuple containing the number of input channels and the number of
            features, by default (6, 60).
        num_classes : int, optional
            Number of output classes, by default 6
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 1e-3
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_channels=input_shape[0])
        self.fc_input_channels = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = self._create_fc(self.fc_input_channels, num_classes)
        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            val_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
            test_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
        )

    def _create_backbone(self, input_channels: int) -> torch.nn.Module:
        return _MultiChannelCNN_HAR(input_channels=input_channels)

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int]
            The input shape of the network.

        Returns
        -------
        int
            The number of features after the convolutional layers.
        """
        random_input = torch.randn(1, 2, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(input_features, num_classes),
            torch.nn.ReLU(),
        )


# def test_multichannel_cnn():
#     from ssl_tools.transforms.signal_1d import FFT, WelchPowerSpectralDensity
#     from ssl_tools.transforms.utils import (
#         PerChannelTransform,
#         StackComposer,
#         Cast,
#     )
#     from ssl_tools.models.utils import RandomDataModule

#     input_shape = (1, 6, 60)

#     fft_transform = PerChannelTransform(FFT(absolute=True))
#     welch = PerChannelTransform(
#         WelchPowerSpectralDensity(
#             fs=1 / 20, return_onesided=False, absolute=True
#         )
#     )
#     stacker = StackComposer([fft_transform, welch])

#     data_module = RandomDataModule(
#         num_samples=8,
#         num_classes=6,
#         input_shape=input_shape,
#         batch_size=8,
#         transforms=[stacker, Cast("float32")],
#     )

#     model = MultiChannelCNN_HAR(
#         input_shape=input_shape, num_classes=6, learning_rate=1e-3
#     )
#     print(model)

#     trainer = L.Trainer(
#         max_epochs=1, logger=False, devices=1, accelerator="cpu"
#     )

#     trainer.fit(model, datamodule=data_module)


# if __name__ == "__main__":
#     test_multichannel_cnn()
