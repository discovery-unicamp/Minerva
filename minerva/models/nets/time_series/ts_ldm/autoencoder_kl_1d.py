import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, Any


def nonlinearity(x):
    """Swish/SiLU activation function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Tensor after applying the activation function sigmoid.
    """
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    """Creates a GroupNorm module with dynamically bounded group size.

    Ensures the number of groups does not exceed the channel count,
    which is critical for temporal/sensor (e.g. HAR) data.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    Returns
    -------
    nn.GroupNorm
        GroupNorm layer with `min(8, in_channels)` groups.
    """
    num_groups = min(8, in_channels)
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class ResnetBlock1d(nn.Module):
    """1D Residual Block featuring pre-activation normalization, dropout, and
    an optional 1x1 convolution shortcut for channel matching.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int, optional
        Number of output channels. If None, defaults to `in_channels`.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    """

    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # First convolutional path
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # Second convolutional path
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # 1x1 Convolution shortcut to match channels if input/output shapes differ
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass of the 1D Residual Block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, L).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C_out, L).
        """

        h = x
        # First layer group: Norm -> Act -> Conv (Pre-activation style)
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Second layer group: Norm -> Act -> Dropout -> Conv
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Apply 1x1 conv shortcut if there is a channel mismatch
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        # Residual connection
        return x + h


class Downsample1d(nn.Module):
    """1D Downsampling layer using strided convolution with asymmetric padding.

    Parameters
    ----------
    in_channels : int
        Number of input and output channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        """Forward pass for 1D downsampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Downsampled output tensor of shape (B, C, L // 2).
        """
        # Asymmetric padding to maintain parity with original SD behavior
        x = F.pad(x, (0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D Upsampling layer using nearest-neighbor interpolation followed by convolution.

    Parameters
    ----------
    in_channels : int
        Number of input and output channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward pass for 1D upsampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Upsampled output tensor of shape (B, C, L * 2).
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class AttnBlock1d(nn.Module):
    """1D Self-Attention module applied across the temporal length dimension.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        # 1x1 Convolutions act as linear projections for Q, K, V across the channel dimension
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass for 1D attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Self-attended output tensor of shape (B, C, L).
        """
        h_ = self.norm(x)

        # Q, K, V projections across the channel dimension
        q = self.q(h_).permute(0, 2, 1)
        # k shape: (B, C, L)
        k = self.k(h_)
        # v shape: (B, C, L)
        v = self.v(h_)

        # Attention scores: (B, L, C) x (B, C, L) -> (B, L, L)
        w_ = torch.bmm(q, k)

        # Scale by 1 / sqrt(channels)
        w_ = w_ * (int(k.shape[1]) ** (-0.5))

        # Apply Softmax across the last dimension (keys) and transpose for value multiplication
        w_ = F.softmax(w_, dim=-1).permute(0, 2, 1)

        # Attend to values: (B, C, L) x (B, L, L) -> (B, C, L)
        h_ = torch.bmm(v, w_)

        # Final projection
        h_ = self.proj_out(h_)

        # Residual connection
        return x + h_


class Encoder1d(nn.Module):
    """Hierarchical 1D Encoder network for variational autoencoders.

    Extracts multi-scale temporal features using residual blocks, optional attention,
    and downsampling modules.

    Parameters
    ----------
    ch : int
        Base channel count.
    out_ch : int
        Number of output channels.
    ch_mult : tuple of int, optional
        Channel scale multipliers per level. Default is (1, 2, 4, 8).
    num_res_blocks : int, optional
        Number of residual blocks per level. Default is 2.
    attn_resolutions : tuple of int, optional
        Spatial resolutions at which self-attention is applied. Default is (16,).
    dropout : float, optional
        Dropout probability. Default is 0.0.
    in_channels : int, optional
        Number of input signal channels. Default is 6.
    z_channels : int, optional
        Number of latent channels. Default is 4.
    resolution : int, optional
        Input sequence resolution/length. Default is 64.
    double_z : bool, optional
        Whether to double the output latent channels (for mean and log-variance). Default is True.
    """

    def __init__(
        self,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        in_channels=6,
        z_channels=4,
        resolution=64,
        double_z=True,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Input Layer
        self.conv_in = nn.Conv1d(in_channels, ch, 3, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        # Builds the network hierarchy based on your yaml `ch_mult`
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock1d(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock1d(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1d(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle Blocks
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1d(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock1d(block_in)
        self.mid.block_2 = ResnetBlock1d(block_in, block_in, dropout=dropout)

        # Output Layer
        self.norm_out = Normalize(block_in)
        out_channels_final = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv1d(block_in, out_channels_final, 3, padding=1)

    def forward(self, x):
        """Forward pass of the 1D Encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, resolution).

        Returns
        -------
        torch.Tensor
            Encoded latent tensor of shape (B, 2 * z_channels, L_lat) or (B, z_channels, L_lat).
        """

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder1d(nn.Module):
    """Hierarchical 1D Decoder network for variational autoencoders.

    Reconstructs temporal signals from latent space representations using
    upsampling, residual blocks, and optional self-attention.

    Parameters
    ----------
    ch : int
        Base channel count.
    out_ch : int
        Number of output channels.
    ch_mult : tuple of int, optional
        Channel scale multipliers per level. Default is (1, 2, 4, 8).
    num_res_blocks : int, optional
        Number of residual blocks per level. Default is 2.
    attn_resolutions : tuple of int, optional
        Spatial resolutions at which self-attention is applied. Default is (16,).
    dropout : float, optional
        Dropout probability. Default is 0.0.
    in_channels : int, optional
        Number of input channels. Default is 6.
    z_channels : int, optional
        Number of latent channels. Default is 4.
    resolution : int, optional
        Target signal resolution/length. Default is 64.
    """

    def __init__(
        self,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        in_channels=6,
        z_channels=4,
        resolution=64,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv1d(z_channels, block_in, 3, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1d(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock1d(block_in)
        self.mid.block_2 = ResnetBlock1d(block_in, block_in, dropout=dropout)

        self.up = nn.ModuleList()
        # DYNAMIC REVERSE LOOP: Builds the upsampling path
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1d(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock1d(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1d(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, out_ch, 3, padding=1)

    def forward(self, z):
        """Forward pass of the 1D Decoder.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape (B, z_channels, L_lat).

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor of shape (B, out_ch, resolution).
        """

        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution1d(object):
    """Diagonal Gaussian Distribution wrapper for 1D VAE latent space parameters.

    Parameters
    ----------
    parameters : torch.Tensor
        Concatenated mean and log-variance tensor along channel dimension.
        Shape: (B, 2 * z_channels, L).
    deterministic : bool, optional
        If True, disables stochastic sampling and sets standard deviation to zero.
        Default is False.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters

        # Split the channels in half: the first half is the mean, the second half is the log-variance
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        # Clamp logvar to prevent numerical instability (exploding/vanishing gradients)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic

        # Calculate standard deviation and variance
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        # If deterministic flag is True, turn off the noise by setting variance/std to 0
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        """Draws samples using the reparameterization trick.

        Returns
        -------
        torch.Tensor
            Stochastic latent sample of shape (B, z_channels, L).
        """
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        """Returns the deterministic mean without added noise.

        Returns
        -------
        torch.Tensor
            Mean tensor of shape (B, z_channels, L).
        """
        return self.mean

    def kl(self, other=None):
        """Calculates Kullback-Leibler (KL) divergence against a standard normal
        or another Diagonal Gaussian distribution.

        Parameters
        ----------
        other : DiagonalGaussianDistribution1d, optional
            Target distribution. If None, computes KL divergence against N(0, I).

        Returns
        -------
        torch.Tensor
            KL divergence value summed over space/time dimensions, shape (B,).
        """
        if self.deterministic:
            return torch.Tensor([0.0]).to(self.parameters.device)
        else:
            if other is None:
                # KL divergence against the standard normal distribution N(0, I)
                # Note: dim=[1, 2] is used for 1D sequences [Batch, Channels, Length]
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2]
                )
            else:
                # KL divergence against another custom Gaussian distribution
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2],
                )


class AutoencoderKL1d(L.LightningModule):
    """1D Variational Autoencoder with KL Divergence penalty adapted for
    Human Activity Recognition (HAR) and time-series sensor signals.

    Parameters
    ----------
    ddconfig : Dict[str, Any]
        Dictionary configuration containing architectural parameters for
        `Encoder1d` and `Decoder1d`.
    embed_dim : int, optional
        Latent embedding dimension. Default is 4.
    lr : float, optional
        Learning rate for the optimizer. Default is 4.5e-4.
    kl_weight : float, optional
        Weight for the KL divergence term in total loss. Default is 1.0e-6.
    original_length : int, optional
        Unpadded target signal length. Default is 60.
    """

    def __init__(
        self,
        ddconfig: Dict[str, Any],
        embed_dim=4,
        lr=4.5e-4,
        kl_weight=1.0e-6,
        original_length=60,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.kl_weight = kl_weight
        self.embed_dim = embed_dim

        # Save sequence size configurations
        self.original_length = original_length
        self.padded_length = ddconfig["resolution"]  # Expected to be 64 for HAR

        # Dynamic construction based on your configuration (Same as SD model)
        self.encoder = Encoder1d(
            ch=ddconfig["ch"],
            out_ch=ddconfig["out_ch"],
            ch_mult=ddconfig["ch_mult"],
            num_res_blocks=ddconfig["num_res_blocks"],
            attn_resolutions=ddconfig["attn_resolutions"],
            dropout=ddconfig["dropout"],
            in_channels=ddconfig["in_channels"],
            z_channels=ddconfig["z_channels"],
            resolution=ddconfig["resolution"],
            double_z=ddconfig[
                "double_z"
            ],  # We want mean and logvar for the latent distribution
        )

        self.decoder = Decoder1d(
            ch=ddconfig["ch"],
            out_ch=ddconfig["out_ch"],
            ch_mult=ddconfig["ch_mult"],
            num_res_blocks=ddconfig["num_res_blocks"],
            attn_resolutions=ddconfig["attn_resolutions"],
            dropout=ddconfig["dropout"],
            in_channels=ddconfig["in_channels"],
            z_channels=ddconfig["z_channels"],
            resolution=ddconfig["resolution"],
        )

        # Post-encoder / Pre-decoder convolutions
        self.quant_conv = nn.Conv1d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)

    def adapter_pad(self, x):
        """Pads sequence to model resolution using replication to preserve continuity.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, original_length).

        Returns
        -------
        torch.Tensor
            Padded tensor of shape (B, C, padded_length).
        """
        if x.shape[-1] == self.original_length:
            pad_size = self.padded_length - self.original_length
            x = F.pad(x, (0, pad_size), mode="replicate")
        return x

    def adapter_unpad(self, x):
        """Crops padded sequence back to its original physical length.

        Parameters
        ----------
        x : torch.Tensor
            Padded tensor of shape (B, C, padded_length).

        Returns
        -------
        torch.Tensor
            Cropped tensor of shape (B, C, original_length).
        """
        if x.shape[-1] == self.padded_length:
            x = x[..., : self.original_length]
        return x

    def encode(self, x):
        """Encodes input signal into latent Gaussian distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (B, C, L).

        Returns
        -------
        DiagonalGaussianDistribution1d
            Distribution parameterizing the latent posterior space.
        """

        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution1d(moments)

    def decode(self, z):
        """Decodes latent representation into reconstructed sequence space.

        Parameters
        ----------
        z : torch.Tensor
            Latent sequence tensor of shape (B, embed_dim, L_lat).

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor of shape (B, C, L).
        """

        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        """Full forward pass consisting of dynamic padding, encoding, sampling,
        decoding, and unpadding.

        Parameters
        ----------
        x : torch.Tensor
            Input batch tensor of shape (B, C, original_length).
        sample_posterior : bool, optional
            Whether to sample stochastically from posterior distribution.
            If False, uses the mean mode. Default is True.

        Returns
        -------
        torch.Tensor
            Final unpadded reconstruction tensor of shape (B, C, original_length).
        DiagonalGaussianDistribution1d
            Latent posterior Gaussian distribution.
        """

        x_padded = self.adapter_pad(x)

        posterior = self.encode(x_padded)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec_padded = self.decode(z)

        dec_final = self.adapter_unpad(dec_padded)
        return dec_final, posterior

    # def get_loss(self, x, x_rec, posterior):
    #     """
    #     Calculates loss ONLY using the original 60-step signal.
    #     The padding area has already been discarded by adapter_unpad.
    #     """
    #     # Reconstruction Loss (L1 + L2 is widely used for time-series / HAR)
    #     rec_loss_l1 = F.l1_loss(x, x_rec, reduction="mean")
    #     rec_loss_l2 = F.mse_loss(x, x_rec, reduction="mean")

    #     fft_x = torch.fft.rfft(x, dim=-1, norm="forward")
    #     fft_rec = torch.fft.rfft(x_rec, dim=-1, norm="forward")
    #     mag_x = torch.abs(fft_x)
    #     mag_rec = torch.abs(fft_rec)

    #     fft_loss = F.l1_loss(mag_x, mag_rec, reduction="mean")

    #     rec_loss = 0.5 * rec_loss_l1 + 0.5 * rec_loss_l2 + 0.2 * fft_loss

    #     # Latent Space KL Divergence
    #     kl_loss = posterior.kl().mean()

    #     # Combined objective function
    #     total_loss = rec_loss + (self.kl_weight * kl_loss)
    #     return total_loss, rec_loss, kl_loss, fft_loss

    def get_loss(self, x, x_rec, posterior):
        """Computes composite objective loss on the original physical sequence length.

        Includes L1 reconstruction loss dynamically weighted by sample energy,
        spectral loss (FFT magnitude), and KL divergence loss.

        Parameters
        ----------
        x : torch.Tensor
            Original ground-truth signal tensor of shape (B, C, original_length).
        x_rec : torch.Tensor
            Reconstructed signal tensor of shape (B, C, original_length).
        posterior : DiagonalGaussianDistribution1d
            Latent Gaussian posterior distribution.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - total_loss : Combined loss value.
            - rec_loss : Reconstruction component loss.
            - kl_loss : KL divergence loss component.
            - fft_loss : Spectral loss component.
        """

        # Unreduced L1 loss calculation to evaluate on a sample-by-sample basis
        rec_loss_l1_raw = F.l1_loss(x, x_rec, reduction="none")  # Forma: (B, C, L)
        rec_loss_l2 = F.mse_loss(x, x_rec, reduction="mean")

        # Dynamic sample weighting by signal energy (standard deviation per sample)
        sample_std = torch.std(x, dim=[1, 2], keepdim=True)  # Forma: (B, 1, 1)

        # Dynamic sample-wise weighting: lower energy samples receive higher weight
        loss_weight = 1.0 / (sample_std + 0.05)

        rec_loss_l1 = (rec_loss_l1_raw * loss_weight).mean()

        # Frequency domain loss (FFT) calculation with dynamic weighting
        fft_x = torch.fft.rfft(x, dim=-1, norm="forward")
        fft_rec = torch.fft.rfft(x_rec, dim=-1, norm="forward")
        mag_x = torch.abs(fft_x)
        mag_rec = torch.abs(fft_rec)

        fft_loss_raw = F.l1_loss(mag_x, mag_rec, reduction="none")
        fft_loss = (fft_loss_raw * loss_weight).mean()

        # Combined reconstruction loss
        rec_loss = 1.0 * rec_loss_l1 + 0.1 * fft_loss + 0.0 * rec_loss_l2

        # Latent Space KL Divergence
        kl_loss = posterior.kl().mean()

        # Total combined loss
        total_loss = rec_loss + (self.kl_weight * kl_loss)
        return total_loss, rec_loss, kl_loss, fft_loss

    def training_step(self, batch, batch_idx):
        """Lightning training step execution.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor
            Input training batch data.
        batch_idx : int
            Index of current batch.

        Returns
        -------
        torch.Tensor
            Computed total training loss for optimization.
        """

        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        x_rec, posterior = self(x)

        loss, rec_loss, kl_loss, fft_loss = self.get_loss(x, x_rec, posterior)

        self.log("train_total_loss", loss, prog_bar=True)
        self.log("train_rec_loss", rec_loss, prog_bar=True)
        self.log("train_kl_loss", kl_loss)
        self.log("train_fft_loss", fft_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Lightning validation step execution.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor
            Input validation batch data.
        batch_idx : int
            Index of current batch.

        Returns
        -------
        torch.Tensor
            Computed total validation loss.
        """

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_rec, posterior = self(x)

        loss, rec_loss, kl_loss, fft_loss = self.get_loss(x, x_rec, posterior)

        self.log("val_total_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_rec_loss", rec_loss, prog_bar=True, sync_dist=True)
        self.log("val_kl_loss", kl_loss, sync_dist=True)
        self.log("val_fft_loss", fft_loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configures the AdamW optimizer for model training.

        Returns
        -------
        torch.optim.Optimizer
            AdamW optimizer instance configured with current parameters and learning rate.
        """

        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
