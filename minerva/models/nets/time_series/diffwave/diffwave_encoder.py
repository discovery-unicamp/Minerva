"""Feature extraction from DiffWave residual blocks."""

import torch
import torch.nn as nn
from typing import Optional
from minerva.models.ssl.diffwave import DiffWave


class DiffWaveEncoder(nn.Module):
    """Extract DiffWave residual features directly or after one reverse diffusion step.

    Single pass extracts features from the input at ``diffusion_timestep``.
    Double pass applies one reverse step with the primary backbone, then extracts
    features at timestep zero with a second backbone initialized from its weights.

    Parameters
    ----------
    backbone : DiffWave
        Initialized DiffWave model, typically loaded with pretrained weights.
    diffusion_timestep : int, optional
        Timestep used by the first pass. Default is 0.
    target_block : int, optional
        Zero-based residual block index at which to extract features. If None,
        uses the last residual block. Default is None.
    pass_strategy : str, optional
        Use ``"double"`` for denoising followed by feature extraction. Other
        values follow the single-pass path. Default is ``"single"``.
    flatten : bool, optional
        If True, flatten the channel and time dimensions. If False, average
        features over time. Default is False.

    Notes
    -----
    The wrapper supplies no class labels and extracts residual activations.
    The reverse step adds Gaussian noise when ``diffusion_timestep > 0``, so
    repeated double-pass calls can produce different representations.
    """

    def __init__(
        self,
        backbone: DiffWave,
        diffusion_timestep: int = 0,
        target_block: Optional[int] = None,
        pass_strategy: str = "single",
        flatten: bool = False,
    ):
        """Configure the timestep, residual block, and feature extraction strategy."""
        super(DiffWaveEncoder, self).__init__()
        self.backbone = backbone
        self.backbone2: DiffWave = None
        self.diffusion_timestep = diffusion_timestep
        self.target_res_layer = target_block
        self.pass_strategy = pass_strategy
        self.flatten = flatten

        self.is_double_pass = self.pass_strategy == "double"
        if self.is_double_pass:
            print("Initializing double pass backbone for DiffWaveEncoder")
            self.backbone2 = DiffWave(**backbone.get_init_config())
            self.backbone2.load_state_dict(backbone.state_dict())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Extract residual features using the configured timestep and block.

        Parameters
        ----------
        input : torch.Tensor
            Signal batch of shape ``(batch_size, in_channels, sequence_length)``.
            The tensor must be on the same device as the backbone.

        Returns
        -------
        torch.Tensor
            Features of shape ``(batch_size, res_channels)`` when ``flatten=False``,
            or ``(batch_size, res_channels * sequence_length)`` when True.
        """
        x = input
        if self.is_double_pass:
            x = self.backbone.full_forward(
                x, None, target_time_step=self.diffusion_timestep
            )
            x = self.backbone2.simple_forward(
                x,
                None,
                target_time_step=0,
                target_res_layer=self.target_res_layer,
                return_skip=False,
                flatten=self.flatten,
            )
        else:
            x = self.backbone.simple_forward(
                x,
                None,
                target_time_step=self.diffusion_timestep,
                target_res_layer=self.target_res_layer,
                return_skip=False,
                flatten=self.flatten,
            )

        return x
