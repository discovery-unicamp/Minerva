from typing import Tuple, Union

import torch

from minerva.models.nets.base import SimpleSupervisedModel


class ZeroPadder2D(torch.nn.Module):
    def __init__(self, pad_at: Tuple[int], padding_size: int):
        super().__init__()
        self.pad_at = pad_at
        self.padding_size = padding_size

    def forward(self, x):
        for i in self.pad_at:
            left = x[:, :, :i, :]
            zeros = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.padding_size,
                x.shape[3],
                device=x.device,
            )
            right = x[:, :, i:, :]

            x = torch.cat([left, zeros, right], dim=2)

        return x

    def __str__(self) -> str:
        return f"ZeroPadder2D(pad_at={self.pad_at}, padding_size={self.padding_size})"

    def __repr__(self) -> str:
        return str(self)


class CNN_HaEtAl_1D_Backbone(torch.nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            # First 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=in_channels,
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

    def forward(self, x):
        # For 1D input, ie,  (batch_size, sensors, time_steps)
        if len(x.shape) == 3:
            # Add a channel dimension: (batch_size, 1, sensors, time_steps)
            x = x.unsqueeze(1)
            x = self.backbone(x)
            # 4D -> 3D
            x = x.flatten(start_dim=2)
            return x
        # For 2D input, ie, (batch_size, channels, sensors, time_steps)
        else:
            return self.backbone(x)


class CNN_HaEtAl_1D(SimpleSupervisedModel):
    def __init__(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]] = (6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        # Arguments passed to the SimpleSupervisedModel constructor
        *args,
        **kwargs,
    ):
        # If input_shape is 2D, add a channel dimension
        if len(input_shape) == 2:
            input_shape = (1, *input_shape)
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = CNN_HaEtAl_1D_Backbone(in_channels=input_shape[0])
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
        # Flatten the output, except for the batch dimension, and get the
        # number of features
        return out.view(out.size(0), -1).size(1)

    def _create_fc(self, input_features: int, num_classes: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )


class CNN_HaEtAl_2D_Backbone(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        pad_at: int = 3,
        first_kernel_size: int = 4,
    ):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            # Add padding
            ZeroPadder2D(
                pad_at=(pad_at,),
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

    def forward(self, x):
        if len(x.shape) == 3:
            # Add a channel dimension: (batch_size, 1, sensors, time_steps)
            x = x.unsqueeze(1)
            x = self.backbone(x)
            # 4D -> 3D
            x = x.flatten(start_dim=2)
            return x
        else:
            return self.backbone(x)


class CNN_HaEtAl_2D(SimpleSupervisedModel):
    def __init__(
        self,
        pad_at: Union[int, Tuple[int]] = (3,),
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]] = (6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        # Arguments passed to the SimpleSupervisedModel constructor
        *args,
        **kwargs,
    ):
        if len(input_shape) == 2:
            input_shape = (1, *input_shape)
        self.pad_at = pad_at if isinstance(pad_at, int) else pad_at[0]
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = CNN_HaEtAl_2D_Backbone(input_shape=input_shape, pad_at=self.pad_at)
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
        in_channels: int = 1,
        pad_at: int = 3,
        out_channels: int = 16,
        include_middle: bool = False,
        permute: bool = False,
        flatten: bool = False,
    ):
        super().__init__()
        self.pad_at = pad_at
        self.in_channels = in_channels
        self.include_middle = include_middle
        self.out_channels = out_channels
        self.first_pad_size = 3 - 1  # kernel -1
        self.permute = permute
        self.flatten = flatten

        self.first_padder = ZeroPadder2D(
            pad_at=(pad_at,),
            padding_size=self.first_pad_size,
        )

        self.upper_part = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
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
                in_channels=in_channels,
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
                    in_channels=in_channels,
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
        if self.permute:
            x = x.permute(0, 2, 1)

        unsqueezed = False
        if len(x.shape) == 3:
            # Add a channel dimension: (batch_size, 1, sensors, time_steps)
            x = x.unsqueeze(1)
            unsqueezed = True

        # X = (batch_size, channels, sensors, time_steps)
        # X = (8, 1, 6, 60)

        # After pad: (8, 1, 8, 60)
        x = self.first_padder(x)

        # upper slice (8, 1, 5, 60)
        upper_x = x[:, :, : self.pad_at + self.first_pad_size, :]
        upper_x = self.upper_part(upper_x)
        zeros_1 = torch.zeros(
            upper_x.size(0),
            upper_x.size(1),
            3 - 1,
            upper_x.size(3),
            device=x.device,
        )

        upper_x = torch.cat(
            [upper_x, zeros_1],
            dim=2,
        )

        # lower slice (8, 1, 5, 60)
        lower_x = x[:, :, self.pad_at :, :]
        lower_x = self.lower_part(lower_x)
        zeros_2 = torch.zeros(
            lower_x.size(0),
            lower_x.size(1),
            3 - 1,
            lower_x.size(3),
            device=x.device,
        )

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

        if unsqueezed:
            result_x = result_x.flatten(start_dim=2)

        if self.flatten:
            result_x = result_x.reshape(x.size(0), -1)

        return result_x


class CNN_PF_2D(SimpleSupervisedModel):
    def __init__(
        self,
        pad_at: int = 3,
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        out_channels: int = 16,
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        include_middle: bool = False,
        # Arguments passed to the SimpleSupervisedModel constructor
        *args,
        **kwargs,
    ):
        self.pad_at = pad_at
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.num_classes = num_classes

        backbone = CNN_PF_Backbone(
            pad_at=pad_at,
            in_channels=input_shape[0],
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
