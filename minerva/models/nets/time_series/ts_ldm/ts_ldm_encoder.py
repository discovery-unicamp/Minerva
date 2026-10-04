import math
import torch
from torch import nn
import copy
from typing import Optional
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.mlp import MLP
from minerva.models.ssl.ts_ldm import TSLatentDiffusion

class TSLatentDiffusionEncoder(nn.Module):
    """Encoder wrapper for TSLatentDiffusion models to extract features.

    Provides a unified interface to extract intermediate representations or
    perform multi-step denoising forward passes from a pre-trained Latent
    Diffusion Model (LDM) tailored for Time Series Analysis.

    Parameters
    ----------
    backbone : nn.Module
        Pre-trained TSLatentDiffusion model used for feature extraction.
    target_time_step : int, optional
        Diffusion timestep used for feature extraction or initial denoising.
        Default is 0.
    target_block : int, optional
        Specific U-Net block index to extract features from. If None, extracts
        from the default output layer. Default is None.
    pass_strategy : str, optional
        Strategy for forward processing iterations. Values of "double" trigger a
        double-pass mechanism (full denoising step followed by feature extraction).
        Default is "single".
    flatten : bool, optional
        Flag indicating whether to flatten the output features. (Note: behavior
        depends on external usage, currently stored as an attribute).
        Default is True.
    """
    def __init__(
        self,
        backbone: nn.Module,
        diffusion_timestep: int = 0,
        target_block: Optional[int] = None,
        pass_strategy: str = "single",
        flatten: bool = True,
    ):
        super(TSLatentDiffusionEncoder, self).__init__()
        self.backbone = backbone
        self.diffusion_timestep = diffusion_timestep
        self.target_block = target_block
        self.pass_strategy = pass_strategy
        self.flatten = flatten
        self.backbone2: TSLatentDiffusion = None
        print(f'Initialized TSLatentDiffusion with pass_strategy={pass_strategy}')     
        
        self.is_double_pass = pass_strategy == "double"
        if self.is_double_pass:
            print("Creating second backbone for double pass... (TSLatentDiffusionEncoder)")
            self.backbone2 = TSLatentDiffusion(**self.backbone.get_init_config())
            self.backbone2.load_state_dict(self.backbone.state_dict())
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract latent features from the input signal.

        If configured for a double pass (`pass_strategy == "double"`), the input first
        undergoes a full single-step denoising using the primary backbone at
        `diffusion_timestep`. The denoised output is then passed through the cloned
        secondary backbone at timestep 0 to extract the block features. 
        Otherwise, it performs a standard feature extraction in a single pass.

        Parameters
        ----------
        input : torch.Tensor
            Input signal tensor to be processed.

        Returns
        -------
        torch.Tensor
            Extracted feature representations.
        """        
        x = input       
        
        if self.is_double_pass :
            x = self.backbone.full_forward( 
                x=x,
                target_time_step=self.diffusion_timestep,
            )
            
            x = self.backbone2.simple_forward(
                x=x,
                target_time_step=0,
                block=self.target_block
            )
        else:
            x = self.backbone.simple_forward(
                x=x,
                target_time_step=self.diffusion_timestep,
                block=self.target_block
            )
            
        return x
