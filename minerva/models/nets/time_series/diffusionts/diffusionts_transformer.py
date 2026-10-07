# This code was copied and adapted from the DiffusionTS implementation in
# https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Models/interpretable_diffusion/transformer.py
import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional
from torch import nn
from einops import rearrange, reduce, repeat
from .diffusionts_model_utils import (
    LearnablePositionalEncoding,
    Conv_MLP,
    AdaLayerNorm,
    Transpose,
    GELU2,
    series_decomp,
)


class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """

    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        """Create polynomial basis functions and learned trend coefficients."""
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1
            ),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1),
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack(
            [lin_space ** float(p + 1) for p in range(trend_poly)], dim=0
        )

    def forward(self, input):
        """Project features onto the polynomial basis to produce the trend."""
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals


class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """

    def __init__(self, out_dim):
        """Choose a moving-average window from the output length."""
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        """Separate the input into residual and moving-average trend."""
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """

    def __init__(self, d_model, low_freq=1, factor=1):
        """Configure the lowest frequency and the number of dominant modes to retain."""
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq : -1]
            f = torch.fft.rfftfreq(t)[self.low_freq : -1]
        else:
            x_freq = x_freq[:, self.low_freq :]
            f = torch.fft.rfftfreq(t)[self.low_freq :]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, "f -> b f d", b=x_freq.size(0), d=x_freq.size(2)).to(
            x_freq.device
        )
        f = rearrange(f[index_tuple], "b f d -> b f () d").to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        """Reconstruct a real temporal signal from selected Fourier coefficients."""
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float), "t -> () () t ()").to(
            x_freq.device
        )

        amp = rearrange(x_freq.abs(), "b f d -> b f () d")
        phase = rearrange(x_freq.angle(), "b f d -> b f () d")
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, "b f t d -> b t d", "sum")

    def topk_freq(self, x_freq):
        """Select the highest-amplitude frequencies for each sample and feature."""
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(
            x_freq.abs(), top_k, dim=1, largest=True, sorted=True
        )
        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing="ij"
        )
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """

    def __init__(self, in_dim, out_dim, factor=1):
        """Build sine and cosine bases and a projection for seasonal coefficients."""
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(
            in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0
        )
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (
            (season_poly // 2, season_poly // 2)
            if season_poly % 2 == 0
            else (season_poly // 2, season_poly // 2 + 1)
        )
        s1 = torch.stack(
            [torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0
        )
        s2 = torch.stack(
            [torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0
        )
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        """Project input features onto the seasonal basis."""
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    """Compute multihead self-attention across the full input sequence."""

    def __init__(
        self,
        n_embd,  # the embed dim
        n_head,  # the number of heads
        attn_pdrop=0.1,  # attention dropout prob
        resid_pdrop=0.1,  # residual attention dropout prob
    ):
        """Build query, key, value, and output projections with dropout."""
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        """Return self-attended features and mean attention weights across heads."""
        B, T, C = x.size()
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    """Attend to encoder features using queries from a decoder sequence."""

    def __init__(
        self,
        n_embd,  # the embed dim
        condition_embd,  # condition dim
        n_head,  # the number of heads
        attn_pdrop=0.1,  # attention dropout prob
        resid_pdrop=0.1,  # residual attention dropout prob
    ):
        """Build decoder queries and encoder key-value projections."""
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        """Return cross-attended decoder features and mean attention weights."""
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(encoder_output)
            .view(B, T_E, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(encoder_output)
            .view(B, T_E, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class EncoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        activate="GELU",
    ):
        """Build timestep-conditioned self-attention and feed-forward layers."""
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, timestep, mask=None, label_emb=None):
        """Apply attention and feed-forward residual updates conditioned on time."""
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))  # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    """Stack timestep-conditioned transformer blocks to encode a series."""

    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        mlp_hidden_times=4,
        block_activate="GELU",
    ):
        """Build the configured number of temporal encoder blocks."""
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times,
                    activate=block_activate,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, input, t, padding_masks=None, label_emb=None):
        """Pass sequence embeddings through all encoder blocks."""
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        activate="GELU",
        condition_dim=1024,
    ):
        """Build attention, trend, seasonality, and feed-forward sublayers."""
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn2 = CrossAttention(
            n_embd=n_embd,
            condition_embd=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        """Return centered features, projected mean, trend, and seasonal components."""
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season


class Decoder(nn.Module):
    """Accumulate trend and seasonal components across transformer decoder blocks."""

    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate="GELU",
        condition_dim=512,
    ):
        """Build decoder blocks conditioned on encoder features."""
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    n_feat=n_feat,
                    n_channel=n_channel,
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times,
                    activate=block_activate,
                    condition_dim=condition_dim,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        """Return decoded features and the accumulated mean, trend, and seasonality."""
        b, c, _ = x.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = self.blocks[block_idx](
                x, enc, t, mask=padding_masks, label_emb=label_emb
            )
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Transformer(nn.Module):
    """Denoise time series by predicting trend and seasonal residual components."""

    def __init__(
        self,
        n_feat: int,
        n_channel: int,
        n_layer_enc: int = 5,
        n_layer_dec: int = 14,
        n_embd: int = 1024,
        n_heads: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        block_activate: str = "GELU",
        max_len: int = 2048,
        conv_params: Optional[tuple] = None,
        **kwargs,
    ):
        """Build feature projections, position embeddings, and encoder-decoder stacks."""
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(
            n_embd,
            n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        self.combine_m = nn.Conv1d(
            n_layer_dec,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="circular",
            bias=False,
        )

        self.encoder = Encoder(
            n_layer_enc,
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            mlp_hidden_times,
            block_activate,
        )
        self.pos_enc = LearnablePositionalEncoding(
            n_embd, dropout=resid_pdrop, max_len=max_len
        )

        self.decoder = Decoder(
            n_channel,
            n_feat,
            n_embd,
            n_heads,
            n_layer_dec,
            attn_pdrop,
            resid_pdrop,
            mlp_hidden_times,
            block_activate,
            condition_dim=n_embd,
        )
        self.pos_dec = LearnablePositionalEncoding(
            n_embd, dropout=resid_pdrop, max_len=max_len
        )

    def forward(self, input, t, padding_masks=None, return_res=False):
        """Return trend and seasonal residual, optionally separating the residual term."""
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)

        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(
            inp_dec, t, enc_cond, padding_masks=padding_masks
        )

        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = (
            self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        )
        trend = self.combine_m(mean) + res_m + trend

        if return_res:
            return (
                trend,
                self.combine_s(season.transpose(1, 2)).transpose(1, 2),
                res - res_m,
            )

        return trend, season_error


if __name__ == "__main__":
    pass
