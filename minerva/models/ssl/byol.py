import copy
import torch
import torchvision
import lightning as L
import numpy as np

from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

from minerva.losses.negative_cossine_similatiry import NegativeCosineSimilarity
from minerva.utils.cossine_scheduler import cosine_schedule
from minerva.utils.deactivate_grad import deactivate_requires_grad

# --- Model Parts ---------------------------------------------------------

# Borrowed from https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/heads.py#L15


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads."""

    def __init__(
        self,
        blocks: Sequence[
            Union[
                Tuple[int, int, Optional[nn.Module], Optional[nn.Module]],
                Tuple[int, int, Optional[nn.Module], Optional[nn.Module], bool],
            ],
        ],
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        for block in blocks:
            input_dim, output_dim, batch_norm, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def preprocess_step(self, x: Tensor) -> Tensor:
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocess_step(x)
        projection: Tensor = self.layers(x)
        return projection


class BYOLProjectionHead(ProjectionHead):
    """Projection head used for BYOL.
    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]
    [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(BYOLProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def preprocess_step(self, x: Tensor) -> Tensor:
        return self.avgpool(x).flatten(start_dim=1)


class BYOLPredictionHead(ProjectionHead):
    """Prediction head used for BYOL.
    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]
    [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
    """

    def __init__(
        self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(BYOLPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


# --- Class implementation ----------------------------------------------------------


class BYOL(L.LightningModule):
    def __init__(
        self, 
        backbone:L.LightningModule = None, 
        learning_rate:float = 0.025, 
        schedule:int = 90000
    ):

        super().__init__()
        if backbone:
            self.backbone = backbone
        else:
            resnet = torchvision.models.resnet18()
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.learning_rate = learning_rate
        self.projection_head = BYOLProjectionHead(2048, 4096, 256)
        self.prediction_head = BYOLPredictionHead(256, 4096, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()
        self.schedule_length = schedule

    def forward(self, x):
        y = self.backbone(x)
        if isinstance(y, OrderedDict):
            y = y["out"]
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x)
        if isinstance(y, OrderedDict):
            y = y["out"]
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.schedule_length, 0.996, 1)
        self.update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        self.update_momentum(
            self.projection_head, self.projection_head_momentum, m=momentum
        )
        (x0, x1) = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(
            f"train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
        for model_ema, model in zip(model_ema.parameters(), model.parameters()):
            model_ema.data = model_ema.data * m + model.data * (1.0 - m)
