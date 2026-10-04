import torch
import torch.nn as nn
from typing import Optional
from minerva.models.ssl.diffwave import DiffWave

class DiffWaveEncoder(nn.Module):
    """Feature extraction wrapper for DiffWave generative model architectures.

    Provides an interface to extract intermediate activations or residual layer
    representations from a pre-trained DiffWave model, supporting both single-pass
    feature extraction and multi-iteration (double-pass) denoising workflows.

    Parameters
    ----------
    backbone : nn.Module
        Pre-trained DiffWave model instance used as the primary feature extractor.
    diffusion_timestep : int, optional
        Diffusion timestep index used during feature extraction or initial denoising.
        Default is 0.
    target_res_layer : int, optional
        Target residual layer index from which to extract feature representations.
        If None, extracts from the model default. Default is None.
    return_skip : bool, optional
        Whether to return aggregated skip connections instead of residual outputs.
        Default is False.
    pass_strategy : str, optional
        Strategy for forward processing iterations. Values of "double" trigger a
        double-pass mechanism (full denoising followed by timestep 0 feature extraction).
        Default is "single".
    flatten : bool, optional
        Whether to flatten output spatial/temporal dimensions. Default is False.
    """
    def __init__(
        self,
        backbone: DiffWave,       
        diffusion_timestep: int = 0,
        target_block: Optional[int] = None,
        pass_strategy: str = "single",
        flatten: bool = False,
    ):
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
        """Executes the feature extraction forward pass on the input tensor.

        If configured for a double pass (`pass_strategy == "double"`), the input is first
        denoised via a full forward step using the primary backbone at `diffusion_timestep`.
        The output is then passed through the secondary backbone at timestep 0 to extract
        the requested residual or skip layer features. Otherwise, performs direct
        feature extraction in a single pass.

        Parameters
        ----------
        input : torch.Tensor
            Input signal tensor to process through the network.

        Returns
        -------
        torch.Tensor
            Extracted feature representations or denoised tensor.
        """
        x = input
        if self.is_double_pass :
            x = self.backbone.full_forward(x, None, target_time_step=self.diffusion_timestep)
            x = self.backbone2.simple_forward(
                x,
                None,
                target_time_step=0,
                target_res_layer=self.target_res_layer,
                return_skip=False,
                flatten=self.flatten
            )
        else:
            x = self.backbone.simple_forward(
                x,
                None,
                target_time_step=self.diffusion_timestep,
                target_res_layer=self.target_res_layer,
                return_skip=False,
                flatten=self.flatten
            )
            
        return x