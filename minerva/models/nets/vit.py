import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional

import torch
from torch import nn
from torchvision.models.vision_transformer import (
    Conv2dNormActivation,
    ConvStemConfig,
    EncoderBlock,
    _log_api_usage_once,
)


class _Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        aux_output: bool = False,
        aux_output_layers: List[int] | None = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        
        self.aux_output = aux_output
        self.aux_output_layers = aux_output_layers

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding

        if self.aux_output:
            aux_outputs = []
            for i, layer in enumerate(self.layers):
                input = layer(input)
                if i in self.aux_output_layers: # type: ignore
                    aux_outputs.append(self.ln(self.dropout(input)))
            return self.ln(self.dropout(input)), aux_outputs
        

        return self.ln(self.layers(self.dropout(input)))


class _VisionTransformerBackbone(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int | tuple[int, int],
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        aux_output: bool = False,
        aux_output_layers: List[int] | None = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        """
        Initializes a Vision Transformer (ViT) model.

        Parameters
        ----------
        image_size : int or tuple[int, int]
            The size of the input image. If an int is provided, it is assumed
            to be a square image. If a tuple of ints is provided, it represents the height and width of the image.
        patch_size : int
            The size of each patch in the image.
        num_layers : int
            The number of transformer layers in the model.
        num_heads : int
            The number of attention heads in the transformer layers.
        hidden_dim : int
            The dimensionality of the hidden layers in the transformer.
        mlp_dim : int
            The dimensionality of the feed-forward MLP layers in the transformer.
        dropout : float, optional
            The dropout rate to apply. Defaults to 0.0.
        attention_dropout : float, optional
            The dropout rate to apply to the attention weights. Defaults to 0.0.
        num_classes : int, optional
            The number of output classes. Defaults to 1000.
        norm_layer : Callable[..., torch.nn.Module], optional
            The normalization layer to use. Defaults to nn.LayerNorm with epsilon=1e-6.
        conv_stem_configs : List[ConvStemConfig], optional
            The configuration for the convolutional stem layers.
            If provided, the input image will be processed by these convolutional layers before being passed to
            the transformer. Defaults to None.

        """
        super().__init__()
        _log_api_usage_once(self)

        if aux_output:
            assert aux_output_layers is not None
            assert all(
                0 <= i < num_layers for i in aux_output_layers
            ), "Invalid layer index in aux_output_layers"

        if isinstance(image_size, int):
            torch._assert(
                image_size % patch_size == 0, "Input shape indivisible by patch size!"
            )
        elif isinstance(image_size, tuple):
            torch._assert(
                image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0,
                "Input shape indivisible by patch size!",
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.aux_output = aux_output
        self.aux_output_layers = aux_output_layers

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last",
                nn.Conv2d(
                    in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1
                ),
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )

        if isinstance(image_size, int):
            seq_length = (image_size // patch_size) ** 2
        elif isinstance(image_size, tuple):
            seq_length = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = _Encoder(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            aux_output=aux_output,
            aux_output_layers=aux_output_layers,
        )
        self.seq_length = seq_length

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = (
                self.conv_proj.in_channels
                * self.conv_proj.kernel_size[0]
                * self.conv_proj.kernel_size[1]
            )
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(
            self.conv_proj.conv_last, nn.Conv2d
        ):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight,
                mean=0.0,
                std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels),
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

    def _process_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Process the input tensor and return the reshaped tensor and dimensions.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, int, int]: The reshaped tensor, number of rows, and number of columns.
        """
        n, c, h, w = x.shape
        p = self.patch_size

        if isinstance(self.image_size, int):
            torch._assert(
                h == self.image_size,
                f"Wrong image height! Expected {self.image_size} but got {h}!",
            )
            torch._assert(
                w == self.image_size,
                f"Wrong image width! Expected {self.image_size} but got {w}!",
            )
        elif isinstance(self.image_size, tuple):
            torch._assert(
                h == self.image_size[0],
                f"Wrong image height! Expected {self.image_size[0]} but got {h}!",
            )
            torch._assert(
                w == self.image_size[1],
                f"Wrong image width! Expected {self.image_size[1]} but got {w}!",
            )
        else:
            raise ValueError("Invalid image size type!")

        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x, n_h, n_w

    def forward(self, x: torch.Tensor):
        """Forward pass of the Vision Transformer Backbone.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Reshape and permute the input tensor
        x, n_h, n_w = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        if self.aux_output:
            x, aux_outputs = self.encoder(x)
            x = x[:, 1:]
            B, _, C = x.shape
            x = x.reshape(B, n_h, n_w, C).permute(0, 3, 1, 2).contiguous()
            for i, aux_output in enumerate(aux_outputs):
                aux_outputs[i] = aux_output[:, 1:]
                B, _, C = aux_output.shape
                aux_outputs[i] = aux_outputs.reshape(B, n_h, n_w, C).permute(0, 3, 1, 2).contiguous()
            return x, aux_outputs

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 1:]

        B, _, C = x.shape

        x = x.reshape(B, n_h, n_w, C).permute(0, 3, 1, 2).contiguous()

        return x
