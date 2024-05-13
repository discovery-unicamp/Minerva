import torch

from minerva.models.nets.base import SimpleSupervisedModel


class _WiseNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.norm  = nn.BatchNorm3d(1)
        self.conv1 = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = torch.nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = torch.nn.Conv3d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4 = torch.nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool4 = torch.nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv5 = torch.nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv6 = torch.nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv7 = torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv8 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = x.view(
            x.size(0), x.size(1), x.size(3), x.size(4)
        )  # (batch_size, channels, height, width)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        return x


class WiseNet(SimpleSupervisedModel):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        loss_fn: torch.nn.Module = None,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        super().__init__(
            backbone=_WiseNet(in_channels=in_channels, out_channels=out_channels),
            fc=torch.nn.Identity(),
            loss_fn=loss_fn or torch.nn.MSELoss(),
            learning_rate=learning_rate,
            flatten=False,
            **kwargs,
        )

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[:, :, : y.size(2), : y.size(3)]

        loss = self._loss_func(y_hat, y)
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[:, :, : y.size(2), : y.size(3)]
        return y_hat
