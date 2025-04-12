from typing import Tuple

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from minerva.models.nets.base import SimpleSupervisedModel

"""
IMUTransformerEncoder model
"""


class _IMUTransformerEncoder(nn.Module):

    def __init__(
        self,
        input_shape: tuple = (6, 60),
        transformer_dim: int = 64,
        encode_position: bool = True,
        nhead: int = 8,
        dim_feedforward: int = 128,
        transformer_dropout: float = 0.1,
        transformer_activation: str = "gelu",
        num_encoder_layers: int = 6,
        permute: bool = False,
    ):
        """
        input_shape: (tuple) shape of the input data
        transformer_dim: (int) dimension of the transformer
        encode_position: (bool) whether to encode position or not
        nhead: (int) number of attention heads
        dim_feedforward: (int) dimension of the feedforward network
        transformer_dropout: (float) dropout rate for the transformer
        transformer_activation: (str) activation function for the transformer
        num_encoder_layers: (int) number of transformer encoder layers
        num_classes: (int) number of output classes
        permute: bool, optional. If `True` the input data will be permuted before passing through the model, by default False.
        """
        super().__init__()

        self.input_shape = input_shape
        self.transformer_dim = transformer_dim
        self.permute = permute

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_shape[0], self.transformer_dim, 1),
            nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1),
            nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1),
            nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1),
            nn.GELU(),
        )

        self.encode_position = encode_position
        encoder_layer = TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(self.transformer_dim),
        )
        self.cls_token = nn.Parameter(
            torch.zeros((1, self.transformer_dim)), requires_grad=True
        )

        if self.encode_position:
            self.position_embed = nn.Parameter(
                torch.randn(input_shape[1] + 1, 1, self.transformer_dim)
            )

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """Forward

        Parameters
        ----------
        x : _type_
            A tensor of shape (B, C, S) with B = batch size, C = channels, S = sequence length

        """
        if self.permute:
            x = x.permute(0, 2, 1)
        # Embed in a high dimensional space and reshape to Transformer's expected shape
        x = self.input_proj(x)
        # print(f"src.shape: {src.shape}")
        x = x.permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([cls_token, x])

        # Add the position embedding
        if self.encode_position:
            x += self.position_embed

        # Transformer Encoder pass
        target = self.transformer_encoder(x)[0]

        return target


class IMUTransformerEncoder(SimpleSupervisedModel):
    def __init__(
        self,
        input_shape: tuple = (6, 60),
        transformer_dim: int = 64,
        encode_position: bool = True,
        nhead: int = 8,
        dim_feedforward: int = 128,
        transformer_dropout: float = 0.1,
        transformer_activation: str = "gelu",
        num_encoder_layers: int = 6,
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        # Arguments passed to the SimpleSupervisedModel constructor
        *args,
        **kwargs,
    ):
        self.input_shape = input_shape

        backbone = self._create_backbone(
            input_shape=input_shape,
            transformer_dim=transformer_dim,
            encode_position=encode_position,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            transformer_dropout=transformer_dropout,
            transformer_activation=transformer_activation,
            num_encoder_layers=num_encoder_layers,
        )

        fc = self._create_fc(transformer_dim, num_classes)

        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            loss_fn=torch.nn.CrossEntropyLoss(),
            *args,
            **kwargs,
        )

    def _create_backbone(
        self,
        input_shape,
        transformer_dim,
        encode_position,
        nhead,
        dim_feedforward,
        transformer_dropout,
        transformer_activation,
        num_encoder_layers,
    ):
        backbone = _IMUTransformerEncoder(
            input_shape=input_shape,
            transformer_dim=transformer_dim,
            encode_position=encode_position,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            transformer_dropout=transformer_dropout,
            transformer_activation=transformer_activation,
            num_encoder_layers=num_encoder_layers,
        )
        return backbone

    def _create_fc(self, transform_dim, num_classes):
        imu_head = nn.Sequential(
            nn.LayerNorm(transform_dim),
            nn.Linear(transform_dim, transform_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(transform_dim // 4, num_classes),
        )
        return imu_head


class IMUCNN(SimpleSupervisedModel):
    def __init__(
        self,
        input_shape: tuple = (6, 60),
        hidden_dim: int = 64,
        num_classes: int = 6,
        dropout_factor: float = 0.1,
        learning_rate: float = 1e-3,
        # Arguments passed to the SimpleSupervisedModel constructor
        *args,
        **kwargs,
    ):
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.dropout_factor = dropout_factor

        backbone = self._create_backbone(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            dropout_factor=dropout_factor,
        )
        self.fc_input_channels = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = self._create_fc(self.fc_input_channels, hidden_dim, num_classes)

        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            loss_fn=torch.nn.CrossEntropyLoss(),
            flatten=True,
            *args,
            **kwargs,
        )

    def _create_backbone(self, input_shape, hidden_dim, dropout_factor):
        return torch.nn.Sequential(
            torch.nn.Conv1d(input_shape[0], hidden_dim, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_factor),
            torch.nn.MaxPool1d(kernel_size=2),
        )

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int]
    ) -> int:
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def _create_fc(self, input_features, hidden_dim, num_classes):
        return torch.nn.Sequential(
            torch.nn.Linear(input_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )
