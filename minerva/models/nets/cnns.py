from typing import List, Tuple

import torch
from torchmetrics import Accuracy

from minerva.models.nets.base import SimpleSupervisedModel


class ZeroPadder2D(torch.nn.Module):
    def __init__(self, pad_at: List[int], padding_size: int):
        super().__init__()
        self.pad_at = pad_at
        self.padding_size = padding_size

    def forward(self, x):

        for i in self.pad_at:
            left = x[:, :, :i, :]
            zeros = torch.zeros(x.shape[0], x.shape[1], self.padding_size, x.shape[3])
            right = x[:, :, i:, :]

            x = torch.cat([left, zeros, right], dim=2)

        return x

    def __str__(self) -> str:
        return f"ZeroPadder2D(pad_at={self.pad_at}, padding_size={self.padding_size})"

    def __repr__(self) -> str:
        return str(self)


class CNN_HaEtAl_1D(SimpleSupervisedModel):

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_shape=input_shape)
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
            val_metrics={"acc": Accuracy(task="multiclass", num_classes=num_classes)},
            test_metrics={"acc": Accuracy(task="multiclass", num_classes=num_classes)},
        )

    def _create_backbone(self, input_shape: Tuple[int, int]) -> torch.nn.Module:
        return torch.nn.Sequential(
            # First 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=32,
                kernel_size=(1, 4),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3),
            ),
            # Second 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 5),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3),
            ),
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

    def _create_fc(self, input_features: int, num_classes: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )


class CNN_HaEtAl_2D(SimpleSupervisedModel):
    def __init__(
        self,
        pad_at: List[int] = (3,),
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        self.pad_at = pad_at
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_shape=input_shape)
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
            val_metrics={"acc": Accuracy(task="multiclass", num_classes=num_classes)},
            test_metrics={"acc": Accuracy(task="multiclass", num_classes=num_classes)},
        )

    def _create_backbone(self, input_shape: Tuple[int, int]) -> torch.nn.Module:
        first_kernel_size = 4
        return torch.nn.Sequential(
            # Add padding
            ZeroPadder2D(
                pad_at=self.pad_at,
                padding_size=first_kernel_size - 1,  # kernel size - 1
            ),
            # First 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=32,
                kernel_size=(first_kernel_size, first_kernel_size),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(3, 3),
                padding=1,
            ),
            # Second 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(3, 3),
                padding=1,
            ),
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

    def _create_fc(self, input_features: int, num_classes: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )


class CNN_PF_Backbone(torch.nn.Module):
    def __init__(
        self,
        pad_at: int,
        input_shape: Tuple[int, int, int],
        out_channels: int = 16,
        include_middle: bool = False,
    ):
        super().__init__()
        self.pad_at = pad_at
        self.input_shape = input_shape
        self.include_middle = include_middle
        self.out_channels = out_channels
        self.first_pad_size = 3 - 1  # kernel -1

        self.first_padder = ZeroPadder2D(
            pad_at=(pad_at,),
            padding_size=self.first_pad_size,
        )

        self.upper_part = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.input_shape[0],
                out_channels=self.out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(2, 3),
                stride=(2, 3),
                padding=1,
            ),
        )

        self.lower_part = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.input_shape[0],
                out_channels=self.out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(2, 3),
                stride=(2, 3),
                padding=1,
            ),
        )

        if self.include_middle:
            self.middle_part = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.input_shape[0],
                    out_channels=self.out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(
                    kernel_size=(2, 3),
                    stride=(2, 3),
                    padding=1,
                ),
            )

        self.shared_part = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=(
                    self.out_channels * 3
                    if self.include_middle
                    else self.out_channels * 2
                ),
                out_channels=64,
                kernel_size=(3, 5),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(2, 3),
                stride=(2, 3),
                padding=1,
            ),
        )

    def forward(self, x):
        # X = (batch_size, channels, sensors, time_steps)
        # X = (8, 1, 6, 60)

        # After pad: (8, 1, 8, 60)
        x = self.first_padder(x)

        # upper slice (8, 1, 5, 60)
        upper_x = x[:, :, : self.pad_at + self.first_pad_size, :]
        upper_x = self.upper_part(upper_x)
        zeros_1 = torch.zeros(upper_x.size(0), upper_x.size(1), 3 - 1, upper_x.size(3))

        upper_x = torch.cat(
            [upper_x, zeros_1],
            dim=2,
        )

        # lower slice (8, 1, 5, 60)
        lower_x = x[:, :, self.pad_at :, :]
        lower_x = self.lower_part(lower_x)
        zeros_2 = torch.zeros(lower_x.size(0), lower_x.size(1), 3 - 1, lower_x.size(3))

        lower_x = torch.cat(
            [zeros_2, lower_x],
            dim=2,
        )

        if self.include_middle:
            # x is already middle
            middle_x = self.middle_part(x)
            concatenated_x = torch.cat([upper_x, middle_x, lower_x], dim=1)

        else:
            concatenated_x = torch.cat([upper_x, lower_x], dim=1)

        result_x = self.shared_part(concatenated_x)
        return result_x


class CNN_PF_2D(SimpleSupervisedModel):
    def __init__(
        self,
        pad_at: int,
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        out_channels: int = 16,
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        include_middle: bool = False,
    ):
        self.pad_at = pad_at
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.num_classes = num_classes

        backbone = CNN_PF_Backbone(
            pad_at=pad_at,
            input_shape=input_shape,
            out_channels=out_channels,
            include_middle=include_middle,
        )
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
            val_metrics={"acc": Accuracy(task="multiclass", num_classes=num_classes)},
            test_metrics={"acc": Accuracy(task="multiclass", num_classes=num_classes)},
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

    def _create_fc(self, input_features: int, num_classes: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=512, out_features=num_classes),
        )


class CNN_PFF_2D(CNN_PF_2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, include_middle=True)
