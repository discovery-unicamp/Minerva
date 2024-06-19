from typing import Callable
import torch
import lightning as L
from minerva.utils.typing import PathLike
from minerva.models.nets.base import SimpleSupervisedModel

class BackboneExtractor:
    def __call__(self, model: SimpleSupervisedModel) -> torch.nn.Module:
        return model.backbone

class FromPretrained(L.LightningModule):
    def __init__(self, model: torch.nn.Module, ckpt_path: PathLike, layer_extractor: Callable = None) -> torch.nn.Module:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict)
        if layer_extractor is not None:
            model = layer_extractor(model)
        print(f"Model loaded from {ckpt_path}!")
        super().__init__()
        self.model = model
        
    def __call__(self, *args, **kwargs) -> torch.nn.Module:
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.model.test_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self.model.predict_step(batch, batch_idx, dataloader_idx)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()