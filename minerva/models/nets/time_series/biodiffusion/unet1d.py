import math
from copy import deepcopy
from functools import partial
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn


# helpers functions
def exists(x):
    return x is not None


def prob_mask_like(shape, prob, device):
    """Generates a boolean tensor mask based on a specified probability.

    Used primarily for condition dropout in classifier-free guidance.

    Parameters
    ----------
    shape : tuple of int or torch.Size
        Shape of the output mask tensor.
    prob : float
        Probability threshold for generating True values (between 0.0 and 1.0).
    device : torch.device
        Hardware device on which to instantiate the output tensor.

    Returns
    -------
    torch.Tensor
        Boolean mask tensor matching `shape`.
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class Residual(nn.Module):
    """Residual connection wrapper module.

    Adds the output of a given function block to its input tensor.

    Parameters
    ----------
    fn : Callable or nn.Module
        Module or function to wrap with a residual shortcut connection.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    """Creates a 1D upsampling module using nearest-neighbor interpolation and convolution.

    Parameters
    ----------
    dim : int
        Number of input channels.
    dim_out : int, optional
        Number of output channels. If None, defaults to `dim`.

    Returns
    -------
    nn.Sequential
        Sequential container with nearest-neighbor upsampling and 1D convolution.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, dim_out if exists(dim_out) else dim, 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """Creates a 1D downsampling module using strided convolution.

    Parameters
    ----------
    dim : int
        Number of input channels.
    dim_out : int, optional
        Number of output channels. If None, defaults to `dim`.

    Returns
    -------
    nn.Conv1d
        1D Convolutional layer with stride 2 for spatial reduction.
    """
    return nn.Conv1d(dim, dim_out if exists(dim_out) else dim, 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv1d):
    """1D Convolutional layer with Weight Standardization.

    Normalizes layer weights during the forward pass to zero mean and unit variance.
    Purportedly works synergistically with Group Normalization (see https://arxiv.org/abs/1903.10520).
    Note: Inherits from `nn.Conv1d` despite the class name.

    Parameters
    ----------
    *args
        Variable length argument list passed to `nn.Conv1d`.
    **kwargs
        Arbitrary keyword arguments passed to `nn.Conv1d`.

    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        """Applies weight standardized 1D convolution to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Convolved output tensor with standardized weights.
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        # mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        # var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        var = weight.var(dim=(1, 2), keepdim=True, unbiased=False)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LayerNorm(nn.Module):
    """Channel-wise Layer Normalization for 1D feature tensors.

    Parameters
    ----------
    dim : int
        Number of feature channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        """Normalizes features across channel dimension with learned affine scaling.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Normalized feature tensor of same shape.
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    """Applies LayerNorm prior to evaluating a wrapped module.

    Parameters
    ----------
    dim : int
        Number of input channels for normalization.
    fn : nn.Module
        Target block to execute after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        """Executes LayerNorm followed by the inner module.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output of wrapped block applied to normalized input.
        """
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for 1D diffusion timesteps.

    Parameters
    ----------
    dim : int
        Output embedding dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Computes sinusoidal positional embeddings for timestep tensor.

        Parameters
        ----------
        x : torch.Tensor
            1D timestep tensor of shape (B,).

        Returns
        -------
        torch.Tensor
            Positional embedding tensor of shape (B, dim).
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """Random or learned Fourier sinusoidal positional embedding module.

    Adapted from crowsonkb's v-diffusion-jax implementation.

    Parameters
    ----------
    dim : int
        Total feature dimension (must be even).
    is_random : bool, optional
        If True, weight parameters remain fixed/frozen. Default is False.
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        """Computes random or learned Fourier embeddings for input tensor.

        Parameters
        ----------
        x : torch.Tensor
            1D input timestep tensor of shape (B,).

        Returns
        -------
        torch.Tensor
            Fourier embedding tensor of shape (B, dim + 1).
        """
        # x = rearrange(x, 'b -> b 1')
        x = x.unsqueeze(-1)
        # freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules
class Block(nn.Module):
    """Convolutional block featuring Weight Standardization, GroupNorm, and SiLU.

    Supports optional scale and shift conditioning features.

    Parameters
    ----------
    dim : int
        Input channel dimension.
    dim_out : int
        Output channel dimension.
    groups : int, optional
        Number of groups for GroupNorm. Default is 8.
    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """Forward pass through convolution, norm, optional modulation, and SiLU.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, C, L).
        scale_shift : tuple of torch.Tensor, optional
            Tuple containing (scale, shift) tensors for adaptive feature modulation. Default is None.

        Returns
        -------
        torch.Tensor
            Processed output tensor of shape (B, dim_out, L).
        """
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """1D ResNet block supporting time and class embedding conditioning.

    Parameters
    ----------
    dim : int
        Input channel dimension.
    dim_out : int
        Output channel dimension.
    time_emb_dim : int, optional
        Time embedding feature dimension. Default is None.
    classes_emb_dim : int, optional
        Class embedding feature dimension. Default is None.
    groups : int, optional
        Number of groups for GroupNorm within internal blocks. Default is 8.
    """

    def __init__(
        self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2),
            )
            if exists(time_emb_dim) or exists(classes_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        """Passes input through conditioned dual blocks with shortcut addition.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, C, L).
        time_emb : torch.Tensor, optional
            Time embedding tensor of shape (B, time_emb_dim). Default is None.
        class_emb : torch.Tensor, optional
            Class embedding tensor of shape (B, class_emb_dim). Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, dim_out, L).
        """
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            # cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            cond_emb = cond_emb.unsqueeze(-1)
            scale_shift = cond_emb.chunk(
                2, dim=1
            )  # split the tensor in two chunks on dim=1

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    """Linear attention mechanism with $O(N)$ memory and time complexity for 1D features.

    Parameters
    ----------
    dim : int
        Input channel dimension.
    heads : int, optional
        Number of attention heads. Default is 4.
    dim_head : int, optional
        Channel dimension per attention head. Default is 32.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        """Applies linear attention across spatial length sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Attention-processed tensor of shape (B, C, L).
        """
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, n), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        # out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        out = out.reshape(b, -1, n)
        return self.to_out(out)


class Attention(nn.Module):
    """Multi-head self-attention mechanism for 1D feature sequences.

    Parameters
    ----------
    dim : int
        Input channel dimension.
    heads : int, optional
        Number of attention heads. Default is 4.
    dim_head : int, optional
        Channel dimension per head. Default is 32.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        """Calculates multi-head self-attention on input tensor sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Self-attention contextualized feature tensor of shape (B, C, L).
        """
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, n), qkv)

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        # out = rearrange(out, 'b h n d -> b (h d) n')
        out = out.transpose(3, 2).reshape(b, -1, n)
        return self.to_out(out)


class Unet1D_cls_free(nn.Module):
    """1D U-Net architecture with Classifier-Free Guidance support.

    Implements a 1-dimensional U-Net designed for diffusion models, incorporating
    both time and class embeddings. It supports condition dropout to seamlessly
    enable classifier-free guidance during generation.

    Parameters
    ----------
    dim : int
        Base channel dimension for the network layers.
    num_classes : int
        Total number of discrete classes for conditional embedding.
    cond_drop_prob : float, optional
        Probability of dropping the class condition (setting it to null) during
        training for classifier-free guidance. Default is 0.5.
    init_dim : int, optional
        Initial convolution output channel dimension. If None, defaults to `dim`.
        Default is None.
    out_dim : int, optional
        Output channel dimension of the final convolution. If None, computed
        based on `channels` and `learned_variance`. Default is None.
    dim_mults : tuple of int, optional
        Channel dimension multipliers for each resolution level. Default is (1, 2, 4, 8).
    channels : int, optional
        Number of input signal channels. Default is 3.
    resnet_block_groups : int, optional
        Number of groups to use for Group Normalization within ResNet blocks.
        Default is 8.
    learned_variance : bool, optional
        Whether the model predicts learned variance (doubles the output channels).
        Default is False.
    learned_sinusoidal_cond : bool, optional
        Whether to use learned sinusoidal positional embeddings. Default is False.
    random_fourier_features : bool, optional
        Whether to use random Fourier features for time conditioning. Default is False.
    learned_sinusoidal_dim : int, optional
        Dimension size for the learned sinusoidal/Fourier embeddings. Default is 16.
    n_timesteps : int, optional
        Total number of diffusion timesteps. Default is 100.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        cond_drop_prob: float = 0.5,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Optional[List[int]] = (1, 2, 4, 8),
        channels: int = 3,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        n_timesteps: int = 100,
    ):
        super().__init__()

        self._init_config = {
            "dim": dim,
            "num_classes": num_classes,
            "cond_drop_prob": cond_drop_prob,
            "init_dim": init_dim,
            "out_dim": out_dim,
            "dim_mults": deepcopy(dim_mults),
            "channels": channels,
            "resnet_block_groups": resnet_block_groups,
            "learned_variance": learned_variance,
            "learned_sinusoidal_cond": learned_sinusoidal_cond,
            "random_fourier_features": random_fourier_features,
            "learned_sinusoidal_dim": learned_sinusoidal_dim,
            "n_timesteps": n_timesteps,
        }

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        self.dim = dim
        self.num_classes = num_classes

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = init_dim if init_dim else dim
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # class embeddings

        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim), nn.GELU(), nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim,
                            groups=resnet_block_groups,
                        ),
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim,
                            groups=resnet_block_groups,
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim,
            classes_emb_dim=classes_dim,
            groups=resnet_block_groups,
        )
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim,
            classes_emb_dim=classes_dim,
            groups=resnet_block_groups,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim,
                            groups=resnet_block_groups,
                        ),
                        ResnetBlock(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim,
                            groups=resnet_block_groups,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv1d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim if out_dim else default_out_dim

        self.final_res_block = ResnetBlock(
            dim * 2,
            dim,
            time_emb_dim=time_dim,
            classes_emb_dim=classes_dim,
            groups=resnet_block_groups,
        )
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        """Executes the forward pass applying classifier-free guidance scaling.

        Evaluates the model both with and without class conditioning, then linearly
        extrapolates between the unconditional and conditional predictions based on
        `cond_scale`.

        Parameters
        ----------
        *args
            Positional arguments passed to the `forward` method (e.g., `x`, `time`, `classes`).
        cond_scale : float, optional
            Guidance scale factor. A value of 1.0 reduces to a standard conditional pass.
            Values > 1.0 push the prediction further in the direction of the condition.
            Default is 1.0.
        **kwargs
            Keyword arguments passed to the `forward` method.

        Returns
        -------
        torch.Tensor
            The scaled output prediction tensor.
        """
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, classes=None, cond_drop_prob=None):
        """Executes the standard full forward pass of the U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (B, C, L).
        time : torch.Tensor
            Diffusion timestep tensor of shape (B,).
        classes : torch.Tensor, optional
            Class label indices of shape (B,). Default is None.
        cond_drop_prob : float, optional
            Probability override for dropping class conditioning. If None, uses the
            instance's default `cond_drop_prob`. Default is None.

        Returns
        -------
        torch.Tensor
            Reconstructed or denoised output tensor of shape (B, out_dim, L).
        """
        batch, device = x.shape[0], x.device

        cond_drop_prob = (
            self.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
        )

        # derive condition, with condition dropout for classifier free guidance

        if classes is not None:
            classes_emb = self.classes_emb(classes)

            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                # null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
                null_classes_emb = self.null_classes_emb.broadcast_to(batch, -1)

                classes_emb = torch.where(
                    # rearrange(keep_mask, 'b -> b 1'),
                    keep_mask.unsqueeze(-1),
                    classes_emb,
                    null_classes_emb,
                )
        else:
            # classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
            classes_emb = self.null_classes_emb.broadcast_to(batch, -1)

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)  # extract embbedings (latent space)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

    def full_forward(self, x, time, classes=None, cond_drop_prob=None):
        """Executes a complete forward pass through the full U-Net network.

        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (B, C, L).
        time : torch.Tensor
            Diffusion timestep tensor of shape (B,).
        classes : torch.Tensor, optional
            Class label indices of shape (B,). Default is None.
        cond_drop_prob : float, optional
            Probability override for dropping class conditioning. If None, uses
            `self.cond_drop_prob`. Default is None.

        Returns
        -------
        torch.Tensor
            Reconstructed/denoised signal tensor of shape (B, out_dim, L).
        """
        batch, device = x.shape[0], x.device

        cond_drop_prob = (
            self.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
        )

        # derive condition, with condition dropout for classifier free guidance

        if classes is not None:
            classes_emb = self.classes_emb(classes)

            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                # null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
                null_classes_emb = self.null_classes_emb.broadcast_to(batch, -1)

                classes_emb = torch.where(
                    # rearrange(keep_mask, 'b -> b 1'),
                    keep_mask.unsqueeze(-1),
                    classes_emb,
                    null_classes_emb,
                )
        else:
            # classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
            classes_emb = self.null_classes_emb.broadcast_to(batch, -1)

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)  # extract embbedings (latent space)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

    def simple_forward(
        self, x, time, classes=None, cond_drop_prob=None, target_block=None
    ):
        """Executes a partial forward pass to extract latent features from a target block.

        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (B, C, L).
        time : torch.Tensor
            Diffusion timestep tensor of shape (B,).
        classes : torch.Tensor, optional
            Class label indices of shape (B,). Default is None.
        cond_drop_prob : float, optional
            Probability override for dropping class conditioning. Default is None.
        target_block : int, optional
            Index of the target block up to which feature extraction is performed.
            If matched during downsampling or mid-block processing, early stops and
            returns the feature map. Default is None.

        Returns
        -------
        torch.Tensor or None
            Intermediate feature embedding tensor extracted at the specified target block.
        """
        batch, device = x.shape[0], x.device

        cond_drop_prob = (
            self.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
        )

        # derive condition, with condition dropout for classifier free guidance

        if classes is not None:
            classes_emb = self.classes_emb(classes)

            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                # null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
                null_classes_emb = self.null_classes_emb.broadcast_to(batch, -1)

                classes_emb = torch.where(
                    # rearrange(keep_mask, 'b -> b 1'),
                    keep_mask.unsqueeze(-1),
                    classes_emb,
                    null_classes_emb,
                )
        else:
            # classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
            classes_emb = self.null_classes_emb.broadcast_to(batch, -1)

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        cont = 1
        emb = None
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)  # extract embbedings (latent space)
            emb = x
            if (target_block is not None) and (cont == target_block):
                break
            cont += 1

        if (len(self.downs) + 1) == target_block:
            x = self.mid_block1(x, t, c)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t, c)
            emb = x

        return emb

    def get_init_config(self):
        """Return all constructor arguments used to initialize the model.

        Returns
        -------
        dict
            Complete constructor configuration.
        """
        return deepcopy(self._init_config)
