import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# This implementation is based and addapted from Fudan Zhang Vision Group SETR implementation.
# You can find the original implementation here: https://github.com/fudan-zvg/SETR/blob/main/mmseg/models/backbones/vit.py#L3


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer: partial[nn.LayerNorm] | nn.LayerNorm = partial(nn.LayerNorm),
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768) -> None:
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        img_size=224,
        feature_size=None,
        in_chans=3,
        embed_dim=768,
    ) -> None:
        super().__init__()
        assert isinstance(backbone, nn.Module), "backbone must be nn.Module"
        self.backbone = backbone
        self.img_size = (img_size, img_size)

        # FIXME (from original code) this is hacky, but most reliable way of determining the exact dim of the output feature
        # map for all networks, the feature metadata has reliable channel and stride info, but using
        # stride to calc feature dim requires info about padding of each stage that isn't captured.
        if feature_size is None:
            with torch.no_grad():
                training = self.backbone.training
                if training:
                    self.backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_chans, self.img_size[0], self.img_size[1])
                )[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]

        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model implementation.

    Parameters
    ----------
        img_size: int
            Size of the input image. Default is 384.
        patch_size: int
            Size of the image patch. Default is 16.
        in_chans: int
            Number of input channels. Default is 3.
        embed_dim: int
            Dimensionality of the token embeddings. Default is 1024.
        depth: int
            Number of transformer blocks. Default is 24.
        num_heads: int
            Number of attention heads. Default is 16.
        num_classes: int
            Number of output classes. Default is 19.
        mlp_ratio: float
            Ratio of MLP hidden dimension to embedding dimension. Default is 4.0.
        qkv_bias: bool
            Whether to include bias in the query, key, and value projections. Default is True.
        qk_scale: float
            Scale factor for query and key. Default is None.
        drop_rate: float
            Dropout rate. Default is 0.1.
        attn_drop_rate: float
            Dropout rate for attention weights. Default is 0.0.
        drop_path_rate: float
            Dropout rate for stochastic depth. Default is 0.0.
        hybrid_backbone: None | nn.Module
            Hybrid backbone module. Default is None.
        norm_layer: nn.Module
            Normalization layer. Default is nn.LayerNorm with eps=1e-6.
        norm_cfg: None | dict
            Normalization configuration. Default is None.
        pos_embed_interp: bool
            Whether to interpolate positional embeddings. Default is False.
        random_init: bool
            Whether to initialize weights randomly. Default is False.
        align_corners: bool
            Whether to align corners in positional embeddings. Default is False.

    References
    ----------
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
        https://arxiv.org/abs/2010.11929

    """

    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=19,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        pos_embed_interp=False,
        random_init=False,
        align_corners=False,
        **kwargs,
    ) -> None:
        super(VisionTransformer, self).__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.hybrid_backbone = hybrid_backbone
        self.norm_layer = norm_layer
        self.norm_cfg = norm_cfg
        self.pos_embed_interp = pos_embed_interp
        self.random_init = random_init
        self.align_corners = align_corners

        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

        if self.hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                self.hybrid_backbone,
                img_size=self.img_size,
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
            )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=self.norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        # NOTE (from original code) as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def init_weights(self, pretrained=None) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if not self.random_init:
            raise NotImplementedError("Pretrained model is not supported yet")
        else:
            print("Initialize weight randomly")

    def _conv_filter(self, state_dict, patch_size=16) -> dict:
        """convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if "patch_embed.proj.weight" in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict

    def to_2D(self, x: torch.Tensor) -> torch.Tensor:
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def to_1D(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.reshape(n, c, -1).transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1, 2)

        # originaly credited to Phil Wang
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class MLAHead(nn.Module):

    def build_norm_layer(self, mlahead_channels):
        layer = nn.SyncBatchNorm(mlahead_channels, eps=1e-5)
        for param in layer.parameters():
            param.requires_grad = True
        return layer

    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            self.build_norm_layer(mlahead_channels),
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            self.build_norm_layer(mlahead_channels),
            nn.ReLU(),
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            self.build_norm_layer(mlahead_channels),
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            self.build_norm_layer(mlahead_channels),
            nn.ReLU(),
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            self.build_norm_layer(mlahead_channels),
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            self.build_norm_layer(mlahead_channels),
            nn.ReLU(),
        )
        self.head5 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            self.build_norm_layer(mlahead_channels),
            nn.ReLU(),
            nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            self.build_norm_layer(mlahead_channels),
            nn.ReLU(),
        )

    def forward(self, x2, x3, x4, x5):
        x2 = F.interpolate(
            self.head2(x2),
            4 * x2.shape[-1],
            mode="bilinear",
            align_corners=True,
        )
        x3 = F.interpolate(
            self.head3(x3),
            8 * x3.shape[-1],
            mode="bilinear",
            align_corners=True,
        )
        x4 = F.interpolate(
            self.head4(x4),
            16 * x4.shape[-1],
            mode="bilinear",
            align_corners=True,
        )
        x5 = F.interpolate(
            self.head5(x5),
            32 * x5.shape[-1],
            mode="bilinear",
            align_corners=True,
        )
        return torch.cat([x2, x3, x4, x5], dim=1)
