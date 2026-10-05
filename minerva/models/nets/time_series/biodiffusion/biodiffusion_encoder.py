import math
import torch
from torch import nn
from typing import Optional, List
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.mlp import MLP
from minerva.models.ssl.biodiffusion import BioDiffusion


class BioDiffusionEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        diffusion_timestep: int = 0,
        target_block: Optional[int] = None,
        pass_strategy: str = "single",
        flatten: bool = True,
    ):
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
