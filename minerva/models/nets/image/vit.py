import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import (
    Conv2dNormActivation,
    ConvStemConfig,
    EncoderBlock,
    _log_api_usage_once,
)

from minerva.utils.position_embedding import get_2d_sincos_pos_embed


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
        aux_output_layers: Optional[List[int]] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6
        ),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        self.aux_output = aux_output
        self.aux_output_layers = aux_output_layers

        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )  # from BERT
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
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.pos_embedding

        if self.aux_output:
            aux_outputs = []
            for i, layer in enumerate(self.layers):
                input = layer(input)
                if i in self.aux_output_layers:  # type: ignore
                    aux_outputs.append(self.ln(self.dropout(input)))
            return self.ln(self.dropout(input)), aux_outputs

        return self.ln(self.layers(self.dropout(input)))


class _VisionTransformerBackbone(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        original_resolution: Optional[Tuple[int, int]] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        aux_output: bool = False,
        aux_output_layers: Optional[List[int]] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        """
        Initializes a Vision Transformer (ViT) model.

        Parameters
        ----------
        image_size : int or Tuple[int, int]
            The size of the input image. If an int is provided, it is assumed
            to be a square image. If a tuple of ints is provided, it represents 
            the height and width of the image.
        patch_size : int
            The size of each patch in the image.
        num_layers : int
            The number of transformer layers in the model.
        num_heads : int
            The number of attention heads in the transformer layers.
        hidden_dim : int
            The dimensionality of the hidden layers in the transformer.
        mlp_dim : int
            The dimensionality of the feed-forward MLP layers in the transformer
        original_resolution : Tuple[int, int], optional
            The original resolution of the input image in the pre-training 
            weights. When None, positional embeddings will not be interpolated. 
            Defaults to None.
        dropout : float, optional
            The dropout rate to apply. Defaults to 0.0.
        attention_dropout : float, optional
            The dropout rate to apply to the attention weights. Defaults to 0.0
        num_classes : int, optional
            The number of output classes. Defaults to 1000.
        norm_layer : Callable[..., torch.nn.Module], optional
            The normalization layer to use. Defaults to nn.LayerNorm with 
            epsilon=1e-6.
        conv_stem_configs : List[ConvStemConfig], optional
            The configuration for the convolutional stem layers.
            If provided, the input image will be processed by these 
            convolutional layers before being passed to the transformer. 
            Defaults to None.

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
                image_size % patch_size == 0,
                "Input shape indivisible by patch size!",
            )
        elif isinstance(image_size, tuple):
            torch._assert(
                image_size[0] % patch_size == 0
                and image_size[1] % patch_size == 0,
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
        self.original_resolution = (
            original_resolution if original_resolution else image_size
        )

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
                    in_channels=prev_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
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
            seq_length = (image_size[0] // patch_size) * (
                image_size[1] // patch_size
            )

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
            nn.init.trunc_normal_(
                self.conv_proj.weight, std=math.sqrt(1 / fan_in)
            )
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

    def _process_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Process the input tensor and return the reshaped tensor and dimensions.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, int, int]: The reshaped tensor, number of rows, 
            and number of columns.
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
        x = x.to(torch.float32)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x, n_h, n_w

    def interpolate_pos_embeddings(self, pretrained_pos_embed, new_img_size):
        """Interpolate encoder's positional embeddings to fit a new input size.

        Args:
            pretrained_pos_embed (torch.Tensor): Pretrained positional embeddings.
            new_img_size (Tuple[int, int]): New height and width of the input image.
        """
        h, w = (
            new_img_size[0] // self.patch_size,
            new_img_size[1] // self.patch_size,
        )
        new_grid_size = (h, w)

        # Reshape pretrained positional embeddings to match the original grid size

        original_resolution = (
            self.original_resolution
            if isinstance(self.original_resolution, Tuple)
            else (self.original_resolution, self.original_resolution)
        )

        pos_embed_reshaped = pretrained_pos_embed[:, 1:].reshape(
            1,
            original_resolution[0] // self.patch_size,
            original_resolution[1] // self.patch_size,
            -1,
        )

        # Interpolate positional embeddings to the new grid size
        pos_embed_interpolated = (
            F.interpolate(
                pos_embed_reshaped.permute(
                    0, 3, 1, 2
                ),  # (1, C, H, W) for interpolation
                size=new_grid_size,
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, -1, pos_embed_reshaped.shape[-1])
        )

        # Concatenate the CLS token and the interpolated positional embeddings
        cls_token = pretrained_pos_embed[:, :1]
        pos_embed_interpolated = torch.cat(
            (cls_token, pos_embed_interpolated), dim=1
        )

        return pos_embed_interpolated

        return pos_embed_interpolated

    def load_backbone(self, path: str, freeze: bool = False):
        """Loads pretrained weights and handles positional embedding resizing 
        if necessary."""
        # Load the pretrained state dict
        state_dict = torch.load(path)

        # Expected shape for positional embeddings based on current model image size

        image_size = (
            self.image_size
            if isinstance(self.image_size, Tuple)
            else (self.image_size, self.image_size)
        )

        expected_pos_embed_shape = (
            1,
            (image_size[0] // self.patch_size)
            * (image_size[1] // self.patch_size)
            + 1,
            self.hidden_dim,
        )

        # Check if positional embeddings need interpolation
        if (
            state_dict["encoder.pos_embedding"].shape
            != expected_pos_embed_shape
        ):
            # Extract the positional embeddings from the state dict
            pretrained_pos_embed = state_dict["encoder.pos_embedding"]

            # Interpolate to match the current image size
            print(
                "Interpolating positional embeddings to match the new image size."
            )
            with torch.no_grad():
                pos_embed_interpolated = self.interpolate_pos_embeddings(
                    pretrained_pos_embed, (image_size[0], image_size[1])
                )
            state_dict["encoder.pos_embedding"] = pos_embed_interpolated

        # Load the (potentially modified) state dict into the encoder
        self.encoder.load_state_dict(state_dict, strict=False)

        # Optionally freeze parameters
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

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
                B, _, C = aux_outputs[i].shape
                aux_outputs[i] = (
                    aux_outputs[i]
                    .reshape(B, n_h, n_w, C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            return x, aux_outputs

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 1:]

        B, _, C = x.shape

        x = x.reshape(B, n_h, n_w, C).permute(0, 3, 1, 2).contiguous()

        return x

    def load_weights(self, weights_path: str, freeze: bool = False):

        state_dict = torch.load(weights_path)

        # Get expected positional embedding shape based on current image size

        image_size = (
            self.image_size
            if isinstance(self.image_size, Tuple)
            else (self.image_size, self.image_size)
        )

        expected_pos_embed_shape = (
            1,
            (image_size[0] // self.patch_size)
            * (image_size[1] // self.patch_size)
            + 1,
            self.hidden_dim,
        )

        # Check if positional embeddings need interpolation
        if (
            state_dict["encoder.pos_embedding"].shape
            != expected_pos_embed_shape
        ):
            # Extract the positional embeddings from the state dict
            pretrained_pos_embed = state_dict["encoder.pos_embedding"]

            # Interpolate to match the current image size
            print(
                "Interpolating positional embeddings to match the new image size."
            )
            with torch.no_grad():
                pos_embed_interpolated = self.interpolate_pos_embeddings(
                    pretrained_pos_embed, (image_size[0], image_size[1])
                )
            state_dict["encoder.pos_embedding"] = pos_embed_interpolated

        # Load the (potentially modified) state dict
        self.load_state_dict(state_dict, strict=False)

        # Optionally freeze parameters
        if freeze:
            for param in self.parameters():

class MaskedAutoencoderViT(L.LightningModule):
    """
    Masked Autoencoder with VisionTransformer backbone.

    Args:
        img_size (int): Size of input image.
        patch_size (int): Size of image patch.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of token embeddings.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        decoder_embed_dim (int): Dimension of decoder embeddings.
        decoder_depth (int): Number of decoder transformer blocks.
        decoder_num_heads (int): Number of decoder attention heads.
        mlp_ratio (float): Ratio of MLP hidden layer size to embedding size.
        norm_layer (torch.nn.LayerNorm): Normalization layer.
        norm_pix_loss (bool): Whether to normalize pixel loss.

    References:
        - timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
        - DeiT: https://github.com/facebookresearch/deit
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.in_chans = in_chans
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # Initialization
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):  # input: (32, 1, 224, 224)
        """
        Extract patches from input images.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).

        Returns:
            torch.Tensor: Patches of shape (N, num_patches, patch_size^2 * in_chans).
        """
        p = self.patch_embed.patch_size[0]
        assert (
            imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        )  # only square images are supported, and the size must be divisible by the patch size

        h = w = imgs.shape[2] // p
        x = imgs.reshape(
            (imgs.shape[0], self.in_chans, h, p, w, p)
        )  # Transform images into (32, 1, 14, 16, 14, 16)
        x = torch.einsum("nchpwq->nhwpqc", x)  # reshape into (32, 14, 14, 16, 16, 1)
        x = x.reshape(
            (imgs.shape[0], h * w, p**2 * self.in_chans)
        )  # Transform into (32, 196, 256)
        return x

    def unpatchify(self, x):
        """
        Reconstruct images from patches.

        Args:
            x (torch.Tensor): Patches of shape (N, L, patch_size^2 * in_chans).

        Returns:
            torch.Tensor: Reconstructed images of shape (N, C, H, W).
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape((x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L, D).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked input, binary mask, shuffled indices.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Encoded representation, binary mask, shuffled indices.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L, D).
            ids_restore (torch.Tensor): Indices to restore the original order of patches.

        Returns:
            torch.Tensor: Decoded output tensor of shape (N, L, patch_size^2 * in_chans).
        """
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Calculate the loss.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            pred (torch.Tensor): Predicted output of shape (N, L, patch_size^2 * in_chans).
            mask (torch.Tensor): Binary mask of shape (N, L).

        Returns:
            torch.Tensor: Computed loss value.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Forward pass.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Loss value, predicted output, binary mask.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (Tuple[torch.Tensor]): Input batch of images and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss value for the current step.
        """
        imgs, _ = batch
        loss, _, _ = self(imgs)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (Tuple[torch.Tensor]): Input batch of images and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss value for the current step.
        """
        imgs, _ = batch
        loss, _, _ = self(imgs)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self):
        """
        Configure optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Define model architectures

# mae_vit_small_patch16_dec512d8b
# decoder: 512 dim, 8 blocks, depth: 6
mae_vit_small_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=6,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

# mae_vit_base_patch16_dec512d8b
# decoder: 512 dim, 8 blocks,
mae_vit_base_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

# mae_vit_large_patch16_dec512d8b
# decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

# mae_vit_huge_patch14_dec512d8b
# decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = partial(
    MaskedAutoencoderViT,
    patch_size=14,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

# mae_vit_large_patch16_dec256d4b
# decoder: 256 dim, 8 blocks
mae_vit_large_patch16D4d256 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)


# mae_vit_base_patch16_dec256d4b
mae_vit_base_patch16D4d256 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)
