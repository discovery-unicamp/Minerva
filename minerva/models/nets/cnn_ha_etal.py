from typing import List, Tuple
import torch
from torchmetrics import Accuracy

from minerva.models.nets.base import SimpleSupervisedModel


# Implementation of Multi-modal Convolutional Neural Networks for Activity
# Recognition, from Ha, Yu, and Choi.
# https://ieeexplore.ieee.org/document/7379657


class ZeroPadder2D(torch.nn.Module):
    def __init__(self, pad_at: List[int], padding_size: int):
        super().__init__()
        self.pad_at = pad_at
        self.padding_size = padding_size

    def forward(self, x):
        # X = (Batch, channels, H, W)
        # X = (8, 1, 6, 60)

        for i in self.pad_at:
            left = x[:, :, :i, :]
            zeros = torch.zeros(
                x.shape[0], x.shape[1], self.padding_size, x.shape[3]
            )
            right = x[:, :, i:, :]

            x = torch.cat([left, zeros, right], dim=2)
            # print(f"-- Left.shape: {left.shape}")
            # print(f"-- Zeros.shape: {zeros.shape}")
            # print(f"-- Right.shape: {right.shape}")
            # print(f"-- X.shape: {x.shape}")
        
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
            val_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
            test_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
        )

    def _create_backbone(self, input_shape: Tuple[int, int]) -> torch.nn.Module:
        return torch.nn.Sequential(
            # Add padding
            # ZeroPadder2D(
            #     pad_at=self.pad_at,
            #     padding_size=4 - 1,  # kernel size - 1
            # ),
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

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
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
        pad_at: List[int]= (3, ),
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
            val_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
            test_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
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

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )
        
        
        
# def test_cnn_1d():
#     input_shape = (1, 6, 60)

#     data_module = RandomDataModule(
#         num_samples=8,
#         num_classes=6,
#         input_shape=input_shape,
#         batch_size=8,
#     )

#     model = CNN_HaEtAl_1D(
#         input_shape=input_shape, num_classes=6, learning_rate=1e-3
#     )
#     print(model)

#     trainer = L.Trainer(
#         max_epochs=1, logger=False, devices=1, accelerator="cpu"
#     )

#     trainer.fit(model, datamodule=data_module)


# def test_cnn_2d():
#     input_shape = (1, 6, 60)

#     data_module = RandomDataModule(
#         num_samples=8,
#         num_classes=6,
#         input_shape=input_shape,
#         batch_size=8,
#     )

#     model = CNN_HaEtAl_2D(
#         pad_at=[3], input_shape=input_shape, num_classes=6, learning_rate=1e-3
#     )
#     print(model)

#     trainer = L.Trainer(
#         max_epochs=1, logger=False, devices=1, accelerator="cpu"
#     )

#     trainer.fit(model, datamodule=data_module)


# if __name__ == "__main__":
#     import logging
#     logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
#     logging.getLogger("lightning").setLevel(logging.ERROR)
#     logging.getLogger("lightning.pytorch.core").setLevel(logging.ERROR)


#     test_cnn_1d()
#     test_cnn_2d()
