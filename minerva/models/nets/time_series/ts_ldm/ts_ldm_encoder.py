"""Feature extraction from the latent diffusion U-Net for time series."""

import torch
from torch import nn
from typing import Optional
from minerva.models.ssl.ts_ldm import TSLatentDiffusion


class TSLatentDiffusionEncoder(nn.Module):
    """Extract latent U-Net features from a time-series diffusion model.

    Single pass encodes the signal with the first-stage autoencoder and
    extracts U-Net activations at ``diffusion_timestep``. Double pass first
    performs one latent reverse step and decodes it to signal space, then
    extracts features at timestep zero with a second backbone initialized
    from the primary backbone's configuration and weights.

    Parameters
    ----------
    backbone : TSLatentDiffusion
        Initialized latent diffusion model, including its first-stage autoencoder.
    diffusion_timestep : int, optional
        Timestep for direct feature extraction or the initial reverse step.
        Default is 0.
    target_block : int, optional
        Zero-based U-Net input block index at which to return activations.
        If None, uses the output of the last input block. Default is None.
    pass_strategy : str, optional
        Use ``"double"`` for denoising followed by feature extraction. Other
        values follow the single-pass path. Default is ``"single"``.
    flatten : bool, optional
        Stored configuration flag. This wrapper does not flatten or pool the
        returned features. Default is True.

    Notes
    -----
    Encoding samples the first-stage posterior when it supports sampling.
    Double pass also adds reverse-step noise above timestep zero. Evaluation
    mode alone therefore does not guarantee identical features between calls.
    """

    def __init__(
        self,
        backbone: nn.Module,
        diffusion_timestep: int = 0,
        target_block: Optional[int] = None,
        pass_strategy: str = "single",
        flatten: bool = True,
    ):
        """Select a diffusion timestep and feature block, creating a second pass if needed."""
        super(TSLatentDiffusionEncoder, self).__init__()
        self.backbone = backbone
        self.diffusion_timestep = diffusion_timestep
        self.target_block = target_block
        self.pass_strategy = pass_strategy
        self.flatten = flatten
        self.backbone2: TSLatentDiffusion = None
        print(f"Initialized TSLatentDiffusion with pass_strategy={pass_strategy}")

        self.is_double_pass = pass_strategy == "double"
        if self.is_double_pass:
            print(
                "Creating second backbone for double pass... (TSLatentDiffusionEncoder)"
            )
            self.backbone2 = TSLatentDiffusion(**self.backbone.get_init_config())
            self.backbone2.load_state_dict(self.backbone.state_dict())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Encode a signal and return activations from the selected U-Net input block.

        Parameters
        ----------
        input : torch.Tensor
            Signal batch of shape ``(batch_size, channels, sequence_length)``.
            The backbone handles configured padding and latent scaling.

        Returns
        -------
        torch.Tensor
            Activations of shape ``(batch_size, block_channels, block_length)``.
            The selected block and latent resolution determine the feature shape.
            No flattening or temporal pooling is applied.
        """
        x = input

        if self.is_double_pass:
            x = self.backbone.full_forward(
                x=x,
                target_time_step=self.diffusion_timestep,
            )

            x = self.backbone2.simple_forward(
                x=x, target_time_step=0, block=self.target_block
            )
        else:
            x = self.backbone.simple_forward(
                x=x, target_time_step=self.diffusion_timestep, block=self.target_block
            )

        return x
