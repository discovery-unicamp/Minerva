from typing import Tuple

import torch
from torchmetrics import Accuracy

from minerva.models.nets.base import SimpleSupervisedModel


class InceptionModule(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (6, 60),
        stride: int = 1,
        kernel_size: int = 41,
        nb_filters: int = 32,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.stride = stride
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size

        self.build_model()

    def build_model(self):

        # INPUT SHAPE is in format (S, T), time-len and number of sensors (ie, (6, 60))

        ########################################################################
        if self.use_bottleneck and self.input_shape[0] > 1:
            input_inception = torch.nn.Conv1d(
                in_channels=self.input_shape[0],
                out_channels=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                stride=1,
                bias=False,
            )

        else:
            input_inception = torch.nn.Identity()

        ########################################################################
        kernel_size_s = [self.kernel_size // (2**i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                torch.nn.Conv1d(
                    in_channels=self.bottleneck_size,
                    out_channels=self.nb_filters,
                    kernel_size=kernel_size_s[i],
                    padding="same",
                    stride=self.stride,
                    bias=False,
                )
            )

        ########################################################################

        self.max_pool_1 = torch.nn.MaxPool1d(
            kernel_size=3, stride=self.stride, padding=1
        )

        direct_conv = torch.nn.Conv1d(
            in_channels=self.input_shape[0],
            out_channels=self.nb_filters,
            kernel_size=1,
            padding="same",
            stride=1,
            bias=False,
        )

        self.input_inception = input_inception
        self.conv_list = conv_list
        self.direct_conv = direct_conv
        self.batch_norm = torch.nn.BatchNorm1d(self.nb_filters * 4)
        self.activation = torch.nn.ReLU()

    def forward(self, input_tensor):

        input_inception = self.input_inception(input_tensor)
        results = []
        for conv in self.conv_list:
            res = conv(input_inception)
            results.append(res)

        res = self.max_pool_1(input_tensor)
        res = self.direct_conv(res)
        results.append(res)

        x = torch.cat(results, dim=1)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class ShortcutLayer(torch.nn.Module):
    def __init__(self, input_tensor_shape, out_tensor_shape):
        super().__init__()
        self.input_tensor_shape = input_tensor_shape
        self.out_tensor_shape = out_tensor_shape
        self.conv = torch.nn.Conv1d(
            in_channels=self.input_tensor_shape[0],
            out_channels=self.out_tensor_shape[0],
            kernel_size=1,
            padding="same",
        )
        self.batch_norm = torch.nn.BatchNorm1d(self.out_tensor_shape[0])

    def forward(self, input_tensor, output_tensor):
        shortcut_y = self.conv(input_tensor)
        shortcut_y = self.batch_norm(shortcut_y)
        x = torch.add(shortcut_y, output_tensor)
        x = torch.relu(x)
        return x


class _InceptionTime(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (6, 60),
        nb_filters=32,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        depth: int = 6,
        kernel_size: int = 41,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1

        self.build_model()

    def build_model(self):
        random_input = torch.rand(1, *self.input_shape)
        depth_inceptions = []
        shortcuts = []

        x = random_input
        input_res = random_input

        for d in range(self.depth):
            inception = InceptionModule(
                input_shape=x.shape[1:],
                stride=1,
                kernel_size=self.kernel_size,
                nb_filters=self.nb_filters,
                use_bottleneck=self.use_bottleneck,
            )
            depth_inceptions.append(inception)

            # forward pass in inception module
            x = inception(x)

            if self.use_residual and d % 3 == 2:
                shortcut = ShortcutLayer(
                    input_tensor_shape=input_res.shape[1:],
                    out_tensor_shape=x.shape[1:],
                )
                shortcuts.append(shortcut)
                x = shortcut(input_res, x)
                input_res = x

        self.depth_inceptions = depth_inceptions
        self.shortcuts = shortcuts
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        shortcut_no = 0

        input_res = x

        for d, inception in enumerate(self.depth_inceptions):
            x = inception(x)

            if self.use_residual and d % 3 == 2:
                shortcut = self.shortcuts[shortcut_no]
                x = shortcut(input_res, x)
                input_res = x
                shortcut_no += 1

        x = self.global_avg_pool(x)
        x = x.squeeze(2)
        return x


class InceptionTime(SimpleSupervisedModel):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (6, 60),
        nb_filters=32,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        depth: int = 6,
        kernel_size: int = 41,
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        # Arguments passed to the SimpleSupervisedModel constructor
        *args,
        **kwargs,
    ):
        backbone = _InceptionTime(
            input_shape,
            nb_filters,
            use_residual,
            use_bottleneck,
            depth,
            kernel_size,
        )
        self.fc_input_features = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = self._create_fc(self.fc_input_features, num_classes)
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
            torch.nn.Linear(in_features=input_features, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )
