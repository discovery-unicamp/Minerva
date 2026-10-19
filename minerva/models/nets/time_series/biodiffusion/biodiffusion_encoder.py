"""Feature extraction from BioDiffusion U-Net blocks."""

import math
import torch
from torch import nn
from typing import Optional, List
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.mlp import MLP
from minerva.models.ssl.biodiffusion import BioDiffusion


class BioDiffusionEncoder(nn.Module):
    """Extract U-Net features directly or after one BioDiffusion reverse step.

    Single pass extracts features at ``diffusion_timestep``. Double pass first
    applies a reverse step with the primary backbone, then extracts features
    at timestep one with a second backbone initialized from its weights.

    Parameters
    ----------
    backbone : BioDiffusion
        Initialized BioDiffusion model, typically loaded with pretrained weights.
    diffusion_timestep : int, optional
        Timestep for direct feature extraction or the initial reverse step.
        Default is 0.
    target_block : int, optional
        One-based index of the U-Net downsampling block. The index immediately
        after the final downsampling block selects the middle block. If None,
        uses the final downsampling features. Default is None.
    pass_strategy : str, optional
        Use ``"double"`` for denoising followed by feature extraction. Other
        values follow the single-pass path. Default is ``"single"``.
    flatten : bool, optional
        Stored configuration flag. This wrapper does not flatten or pool the
        returned features. Default is True.

    Notes
    -----
    The wrapper supplies no class labels. Reverse steps above timestep zero
    include Gaussian noise, so double-pass features can vary between calls.
    """

    def __init__(
        self,
        backbone: nn.Module,
        diffusion_timestep: int = 0,
        target_block: Optional[int] = None,
        pass_strategy: str = "single",
        flatten: bool = True,
    ):
        """Select the timestep and feature block, cloning the backbone for double pass."""
        super(BioDiffusionEncoder, self).__init__()
        self.backbone = backbone
        self.diffusion_timestep = diffusion_timestep
        self.target_block = target_block
        self.pass_strategy = pass_strategy
        self.flatten = flatten
        self.backbone2: BioDiffusion = None

        self.is_double_pass = self.pass_strategy == "double"
        if self.is_double_pass:
            print("Initializing double pass strategy for BioDiffusionEncoder")
            self.backbone2 = BioDiffusion(**self.backbone.get_init_config())
            self.backbone2.load_state_dict(self.backbone.state_dict())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return activations from the selected U-Net block.

        Parameters
        ----------
        input : torch.Tensor
            Signal batch of shape ``(batch_size, channels, sequence_length)``.
            The tensor must be on the same device as the backbone.

        Returns
        -------
        torch.Tensor
            Activations of shape ``(batch_size, block_channels, block_length)``.
            Channel count and temporal length depend on the selected block.
            No flattening or temporal pooling is applied.
        """
        x = input
        if self.is_double_pass:
            x = self.backbone.full_forward(
                inputs=x, t=self.diffusion_timestep, pass_strategy=self.pass_strategy
            )

            x = self.backbone2.simple_forward(
                inputs=x, t=1, target_block=self.target_block, pass_strategy="single"
            )
        else:
            x = self.backbone.simple_forward(
                inputs=x,
                t=self.diffusion_timestep,
                target_block=self.target_block,
                pass_strategy=self.pass_strategy,
            )

        return x
