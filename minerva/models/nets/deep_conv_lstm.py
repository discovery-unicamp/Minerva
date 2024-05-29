from typing import Tuple
import torch
from torchmetrics import Accuracy


from minerva.models.nets.base import SimpleSupervisedModel


# Implementation of DeepConvLSTM as described in the paper:
# Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable
# Activity Recognition (http://www.mdpi.com/1424-8220/16/1/115/html)


class ConvLSTMCell(torch.nn.Module):
    def __init__(self, input_shape: tuple):
        super().__init__()
        self.input_shape = input_shape
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=64,
                kernel_size=(1, 5),
                stride=(1, 1),
            ),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 5),
                stride=(1, 1),
            ),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 5),
                stride=(1, 1),
            ),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 5),
                stride=(1, 1),
            ),
        )

        self.lstm_input_size = self._calculate_conv_output_shape(
            self.conv_block, input_shape
        )

        self.lstm_1 = torch.nn.LSTM(
            input_size=self.lstm_input_size, hidden_size=128, batch_first=True
        )
        self.lstm_2 = torch.nn.LSTM(
            input_size=128, hidden_size=128, batch_first=True
        )

    def _calculate_conv_output_shape(
        self, backbone, input_shape: Tuple[int, int, int]
    ) -> int:
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        return x


class DeepConvLSTM(SimpleSupervisedModel):
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
            # val_metrics={
            #     "acc": Accuracy(task="multiclass", num_classes=num_classes)
            # },
            # test_metrics={
            #     "acc": Accuracy(task="multiclass", num_classes=num_classes)
            # },
        )

    def _create_backbone(self, input_shape: Tuple[int, int]) -> torch.nn.Module:
        return ConvLSTMCell(input_shape=input_shape)

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
            torch.nn.Linear(
                in_features=input_features, out_features=num_classes
            ),
            torch.nn.ReLU(),
        )


# def test_deep_conv_lstm():
#     input_shape = (1, 6, 60)

#     data_module = RandomDataModule(
#         num_samples=8,
#         num_classes=6,
#         input_shape=input_shape,
#         batch_size=8,
#     )

#     model = DeepConvLSTM(
#         input_shape=input_shape, num_classes=6, learning_rate=1e-3
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
#     test_deep_conv_lstm()
