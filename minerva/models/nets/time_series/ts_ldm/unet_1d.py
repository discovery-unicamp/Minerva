import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


def zero_module(module):
    """Zeros out the parameters of a module and returns it.

    This strictly follows the OpenAI initialization pattern to ensure
    that certain layers initially act as identity mappings.

    Parameters
    ----------
    module : torch.nn.Module
        The module to initialize with zeros.

    Returns
    -------
    torch.nn.Module
        The same module with zeroed parameters.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    """Group normalization with forced float32 precision.

    This version of GroupNorm forces the normalization to run in float32
    even if the model is operating in float16 (Mixed Precision / AMP).
    This prevents numerical overflows and stability issues during training.

    Parameters
    ----------
    num_groups : int
        Number of groups to separate the channels into.
    num_channels : int
        Number of channels expected in the input.
    """

    def forward(self, x):
        """Normalize in float32 and return the original input dtype."""
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """Creates a standard GroupNorm layer with dynamic group sizing.

    Safely computes the number of groups so the model does not crash
    if the number of channels is not perfectly divisible by 32.

    Parameters
    ----------
    channels : int
        Number of input channels.

    Returns
    -------
    GroupNorm32
        A GroupNorm layer cast to float32 for stability.
    """
    groups = 32
    if channels < groups:
        groups = channels
    while channels % groups != 0:
        groups -= 1
    return GroupNorm32(groups, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Generates sinusoidal embeddings for timesteps.

    Parameters
    ----------
    timesteps : torch.Tensor
        A 1-D tensor of N indices, one per batch element.
    dim : int
        The dimension of the output.
    max_period : int, optional
        Controls the minimum frequency of the embeddings. Default is 10000.

    Returns
    -------
    torch.Tensor
        An [N x dim] tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CheckpointFunction(torch.autograd.Function):
    """Custom autograd function for memory-efficient gradient checkpointing.

    Prevents PyTorch's native checkpoint from throwing TypeErrors when
    mixing custom parameter lists.
    """

    @staticmethod
    def forward(ctx, run_function, length, *args):
        """Run without recording activations and retain inputs for recomputation."""
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        """Recompute activations and return input and parameter gradients."""
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """Evaluates a function without caching intermediate activations.

    Parameters
    ----------
    func : callable
        The function or forward pass to evaluate.
    inputs : tuple
        A sequence of positional arguments to pass to `func`.
    params : tuple
        A sequence of parameters `func` depends on.
    flag : bool
        If True, use gradient checkpointing. Otherwise, run normally.

    Returns
    -------
    torch.Tensor
        The output of `func`.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class TimestepBlock(nn.Module):
    """Abstract class for modules that require timestep embeddings."""

    @abstractmethod
    def forward(self, x, emb):
        """Define the interface for layers receiving a timestep embedding."""
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential container that smartly routes timestep embeddings.

    It iterates through its child layers and passes the timestep embedding `emb`
    only to the layers that are instances of `TimestepBlock`.
    """

    def forward(self, x, emb):
        """Pass the timestep embedding only to layers that accept it."""
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample1d(nn.Module):
    """1D Upsampling layer using nearest interpolation and convolution.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    """

    def __init__(self, channels):
        """Create a convolution that preserves channels after upsampling."""
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        """Double the temporal length with nearest-neighbor interpolation and convolve."""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample1d(nn.Module):
    """1D Downsampling layer using strided convolution.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    """

    def __init__(self, channels):
        """Create a stride-two convolution that preserves the channel count."""
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        """Reduce the temporal length using a stride-two convolution."""
        return self.conv(x)


class ResBlock1d(TimestepBlock):
    """1D Residual Block with optional timestep conditioning.

    Features weight normalization, dropout, and dynamic dilation. It follows
    the OpenAI pattern of injecting time embeddings via addition or scale/shift.

    Parameters
    ----------
    channels : int
        Number of input channels.
    emb_channels : int
        Number of timestep embedding channels.
    dropout : float
        Dropout probability.
    out_channels : int, optional
        Number of output channels. If None, defaults to `channels`.
    use_scale_shift_norm : bool, optional
        If True, applies FiLM-like conditioning (scale and shift). Default is False.
    dilation : int, optional
        Dilation rate for the convolutions. Padding is automatically adjusted.
        Default is 1.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dilation=1,
    ):
        """Build a residual block conditioned by a diffusion timestep embedding."""
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dilation = dilation

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(
                channels, self.out_channels, 3, padding=dilation, dilation=dilation
            ),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """Combine convolutional features, timestep conditioning, and the skip path."""
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype).unsqueeze(-1)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock1d(nn.Module):
    """1D Self-Attention Block with gradient checkpointing support.

    Parameters
    ----------
    channels : int
        Number of input channels.
    num_head_channels : int, optional
        Number of channels per attention head. Default is 32.
    use_checkpoint : bool, optional
        If True, uses gradient checkpointing to save memory. Default is False.
    """

    def __init__(self, channels, num_head_channels=32, use_checkpoint=False):
        """Build multihead attention projections and optional activation checkpointing."""
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint

        assert (
            channels % num_head_channels == 0
        ), f"Channels {channels} must be divisible by {num_head_channels}"
        self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        """Apply temporal self-attention, optionally recomputing activations backward."""
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        """Compute scaled query-key attention and add the projected residual."""
        b, c, l = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, l).transpose(2, 3)
        k = k.view(b, self.num_heads, head_dim, l)
        v = v.view(b, self.num_heads, head_dim, l).transpose(2, 3)

        weight = torch.matmul(q, k) * (1.0 / math.sqrt(head_dim))
        weight = F.softmax(weight, dim=-1)

        a = torch.matmul(weight, v)
        a = a.transpose(2, 3).contiguous().view(b, c, l)

        return x + self.proj_out(a)


class UNetModel1d(nn.Module):
    """1D U-Net Model for Time Series Generation / Diffusion.

    This architecture adapts the guided-diffusion U-Net to 1D sequences,
    retaining all timestep routing, residual connections, and attention mechanisms.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input sequence.
    model_channels : int
        Base multiplier for the number of channels in the network.
    out_channels : int
        Number of output channels.
    num_res_blocks : int
        Number of residual blocks per downsample/upsample level.
    attention_resolutions : tuple of int
        A collection of downsample rates at which attention will take place.
        For example, if (2, 4), attention is applied at 1/2 and 1/4 sequence length.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    channel_mult : tuple of int, optional
        Channel multiplier for each level of the U-Net. Default is (1, 2, 4, 8).
    num_head_channels : int, optional
        Number of channels per attention head. Default is 32.
    use_scale_shift_norm : bool, optional
        If True, applies FiLM-like timestep conditioning. Default is False.
    use_checkpoint : bool, optional
        If True, applies gradient checkpointing across the network. Default is False.
    num_classes : int, optional
        If specified, enables class-conditional generation with an embedding layer.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        dropout: float = 0.0,
        channel_mult: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        num_head_channels: int = 32,
        use_scale_shift_norm: bool = False,
        use_checkpoint: bool = False,
        num_classes=None,
    ):
        """Build the temporal U-Net with timestep and optional class conditioning."""
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_head_channels = num_head_channels
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv1d(in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # ====== ENCODER ======
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock1d(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock1d(
                            ch,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample1d(ch)))
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # ====== MIDDLE ======
        self.middle_block = TimestepEmbedSequential(
            ResBlock1d(
                ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm
            ),
            AttentionBlock1d(
                ch, num_head_channels=num_head_channels, use_checkpoint=use_checkpoint
            ),
            ResBlock1d(
                ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm
            ),
        )
        self._feature_size += ch

        # ====== DECODER ======
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock1d(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock1d(
                            ch,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample1d(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # ====== OUTPUT LAYER ======
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv1d(ch, out_channels, 3, padding=1)),
        )

        self._init_config = {
            "num_classes": num_classes,
            "in_channels": in_channels,
            "model_channels": model_channels,
            "out_channels": out_channels,
            "num_res_blocks": num_res_blocks,
            "channel_mult": channel_mult,
            "attention_resolutions": attention_resolutions,
            "num_head_channels": num_head_channels,
            "dropout": dropout,
            "use_checkpoint": use_checkpoint,
            "use_scale_shift_norm": use_scale_shift_norm,
        }

    def forward(self, x, timesteps, y=None):
        """Applies the model to a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            An [N x C x L] tensor of inputs (Batch, Channels, Length).
        timesteps : torch.Tensor
            A 1-D batch of timesteps of shape [N].
        y : torch.Tensor, optional
            An [N] tensor of class labels.

        Returns
        -------
        torch.Tensor
            An [N x out_channels x L] tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        hs = []
        h = x

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)

    def forward_emb(self, x, timesteps, block=None):
        """Returns intermediate activations from the encoder blocks.

        Useful for feature extraction or debugging representations.

        Parameters
        ----------
        x : torch.Tensor
            An [N x C x L] tensor of inputs.
        timesteps : torch.Tensor
            A 1-D batch of timesteps.
        block : int, optional
            The specific input block index to stop at and return the activation.
            If None, returns the output of the entire encoder. Default is None.

        Returns
        -------
        torch.Tensor
            The intermediate activation tensor.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = x
        cont = 0
        # print(f"Size input_blocks : {len(self.input_blocks)}")
        for module in self.input_blocks:
            h = module(h, emb)
            if block is not None and cont == block:
                # print(f"Returning activation from block {block} with shape {h.shape}")
                # print(module)
                return h
            cont += 1
        return h

    def get_init_config(self):
        """Return a copy of the constructor configuration for reconstruction."""
        return self._init_config.copy()
