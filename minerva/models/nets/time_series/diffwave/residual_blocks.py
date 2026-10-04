import torch
import torch.nn as nn
import math
from typing import Optional
from .conv_layers import Conv

class ResidualBlock(nn.Module):
    """Residual block used in the DiffWave architecture.

    Combines dilated convolutions, label conditioning, and diffusion step embeddings.
    Produces both residual and skip connections for hierarchical feature aggregation.

    Parameters
    ----------
    res_channels : int
        Number of channels for residual connections.
    skip_channels : int
        Number of channels for skip connections.
    dilation : int, optional
        Dilation rate for the dilated convolution. Default is 1.
    diffusion_step_embed_dim_out : int, optional
        Dimensionality of the diffusion step embedding. Default is 512.
    """

    def __init__(
        self, res_channels, skip_channels, dilation=1, diffusion_step_embed_dim_out=512
    ):
        super(ResidualBlock, self).__init__()
        self.res_channels = res_channels

        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # dilated conv layer
        self.dilated_conv_layer = Conv(
            self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation
        )

        # residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.parametrizations.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.parametrizations.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)
        self.label_projection = nn.Conv1d(128, 2 * res_channels, kernel_size=1)

    def forward(self, input_data, label_emb=None):
        """Forward pass through the residual block.

        Parameters
        ----------
        input_data : tuple
            A tuple `(x, diffusion_step_embed)`:
            - x: input tensor of shape (B, C, L)
            - diffusion_step_embed: tensor of shape (B, embed_dim)
        label_emb : torch.Tensor, optional
            Optional label embedding tensor of shape (B, 128).

        Returns
        -------
        tuple
            - residual output (for next block)
            - skip output (for aggregation)
        """

        x, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        # add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])
        h = h + part_t

        # dilated conv layer
        h = self.dilated_conv_layer(h)
        if label_emb is not None:
            label_bias = self.label_projection(label_emb.unsqueeze(-1))  # (B, 2C, 1)
            h = h + label_bias

        # gated-tanh nonlinearity
        out = torch.tanh(h[:, : self.res_channels, :]) * torch.sigmoid(
            h[:, self.res_channels :, :]
        )

        # residual and skip outputs
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class ResidualGroup(nn.Module):
    """Group of residual blocks with diffusion step embedding.

    Manages the embedding of diffusion steps and sequentially applies
    multiple dilated residual layers with increasing dilation cycles.

    Parameters
    ----------
    res_channels : int
        Number of residual channels.
    skip_channels : int
        Number of skip channels.
    num_res_layers : int, optional
        Total number of residual layers. Default is 30.
    dilation_cycle : int, optional
        Dilation reset cycle. Default is 10.
    diffusion_step_embed_dim_in : int, optional
        Input dimension for diffusion step embedding. Default is 128.
    diffusion_step_embed_dim_mid : int, optional
        Intermediate embedding dimension. Default is 512.
    diffusion_step_embed_dim_out : int, optional
        Output embedding dimension. Default is 512.
    """

    def __init__(
        self,
        res_channels,
        skip_channels,
        num_res_layers=30,
        dilation_cycle=10,
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
    ):
        super(ResidualGroup, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        # the shared two fc layers for diffusion step embedding
        self.fc_t1 = nn.Linear(
            diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid
        )
        self.fc_t2 = nn.Linear(
            diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out
        )

        # stack all residual blocks with dilations 1, 2, ... , 512, ... , 1, 2, ..., 512
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(
                ResidualBlock(
                    res_channels,
                    skip_channels,
                    dilation=2 ** (n % dilation_cycle),
                    diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                )
            )

    def forward(self, input_data, label_emb=None):
        x, diffusion_steps = input_data

        # embed diffusion step t
        diffusion_step_embed = self.calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_in, device=x.device
        )
        diffusion_step_embed = self.swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = self.swish(self.fc_t2(diffusion_step_embed))

        # pass all residual layers
        h = x
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n](
                (h, diffusion_step_embed), label_emb
            )  # use the output from last residual layer
            skip = skip + skip_n  # accumulate all skip outputs

        return skip * math.sqrt(
            1.0 / self.num_res_layers
        )  # normalize for training stability

    def forward_emb(
        self, input_data, label_emb=None, target_res_layer: Optional[int] = None
    ):
        x, diffusion_steps = input_data

        # embed diffusion step t
        diffusion_step_embed = self.calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_in, device=x.device
        )
        diffusion_step_embed = self.swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = self.swish(self.fc_t2(diffusion_step_embed))

        # pass all residual layers
        h = x
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n](
                (h, diffusion_step_embed), label_emb
            )  # use the output from last residual layer
            skip = skip + skip_n  # accumulate all skip outputs
            if target_res_layer is not None and n == target_res_layer:
                return h, skip
        return h, skip

    def calc_diffusion_step_embedding(
        self, diffusion_steps, diffusion_step_embed_dim_in, device="cuda"
    ):
        """
        Embed a diffusion step $t$ into a higher dimensional space
        E.g. the embedding vector in the 128-dimensional space is
        [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

        Parameters:
        diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                    diffusion steps for batch data
        diffusion_step_embed_dim_in (int, default=128):
                                    dimensionality of the embedding space for discrete diffusion steps

        Returns:
        the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
        """

        assert diffusion_step_embed_dim_in % 2 == 0

        half_dim = diffusion_step_embed_dim_in // 2
        _embed = math.log(10000) / (half_dim - 1)
        # _embed = torch.exp(torch.arange(half_dim) * -_embed).to(device)
        _embed = torch.exp(torch.arange(half_dim, device=device) * -_embed)
        _embed = diffusion_steps * _embed
        diffusion_step_embed = torch.cat((torch.sin(_embed), torch.cos(_embed)), 1)

        return diffusion_step_embed

    def swish(self, x):
        return x * torch.sigmoid(x)