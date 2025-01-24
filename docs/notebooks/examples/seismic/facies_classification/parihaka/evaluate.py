#!/usr/bin/env python3

import numpy as np
import torch
from minerva.models.loaders import FromPretrained
from typing import Tuple
from pathlib import Path
from common import get_data_module
import math
from minerva.engines.patch_inferencer_engine import PatchInferencer
import lightning as L
from finetuned_models import (
    byol,
    fastsiam,
    kenshodense,
    simclr,
    tribyol,
    sam,
    lfr,
    dinov2_dpt,
    dinov2_mla,
    dinov2_pup,
    sfm_base_patch16,
)
import traceback
from typing import Tuple

root_data_dir = Path(
    "/workspaces/HIAAC-KR-Dev-Container/shared_data/seam_ai_datasets/seam_ai/images"
)
root_annotation_dir = Path(
    "/workspaces/HIAAC-KR-Dev-Container/shared_data/seam_ai_datasets/seam_ai/annotations"
)

predictions_path = Path.cwd() / "predictions"


class PredictorWrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Adds a dimension after logits to match PatchInferencer requirements
        # 6x1006x590 -> 6x1x1006x590
        res = self.model(x)
        res = res.unsqueeze(2)
        return res


class PatchedPredictorWrapper(L.LightningModule):
    def __init__(self, model: PatchInferencer):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, batch, batch_idx):
        if not isinstance(batch, torch.Tensor) and len(batch) == 2:
            # print("::Batch is a tuple, unpacking it and passing to patch inferencer")
            batch_x, batch_y = batch
            outputs = self.forward(batch_x)
            # return outputs, batch_y
            return outputs
        else:
            # print("::Batch is not a tuple, passing to patch inferencer normal")
            return self.forward(batch)


def load_model(model, ckpt):
    return FromPretrained(model, ckpt, strict=False)


def load_model_from_info(model_info):
    model = model_info["model"]
    ckpt_file = model_info["ckpt_file"]
    return load_model(model, ckpt_file)


def load_model_and_data_module(
    model_instantiator_func,
    img_shape: Tuple[int, int] = (1006, 590),
    n_classes: int = 6,
    batch_size: int = 1,
):
    # Model Info is a dictionary containing information about the model:
    #   name: str
    #   model: L.LightningModule
    #   ckpt_file: Path
    #   img_size: Tuple[int, int]
    #   single_channel: bool
    model_info = model_instantiator_func()

    # ---- 1. Data ----
    data_module = get_data_module(
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        img_size=None,  # Uses original image size (no resize)
        single_channel=model_info["single_channel"],  # 1 or 3 channels
        batch_size=batch_size,
        seed=42,
        num_workers=12
    )

    # ---- 2. Model and wrapper ----

    # Let's check if padding is needed.
    # If shape of model and data is the same, no padding is needed
    if img_shape == model_info["img_size"]:
        pad_dict = None
    else:
        model_h, model_w = model_info["img_size"]
        h_ratio = math.ceil(img_shape[0] / model_h)
        w_ratio = math.ceil(img_shape[1] / model_w)
        pad_dict = {
            "mode": "constant",
            "value": 0,
            "pad": (0, h_ratio * model_h, w_ratio * model_w),
        }

    model_input_shape = (
        1 if model_info["single_channel"] else 3,
        *model_info["img_size"],
    )
    model_output_shape = (n_classes, 1, *model_info["img_size"])

    # Load model
    model = load_model_from_info(model_info)
    model = PredictorWrapper(model)
    model = PatchInferencer(
        model=model,  # type: ignore (as used only for inferencing)
        input_shape=model_input_shape,
        output_shape=model_output_shape,
        padding=pad_dict.copy() if pad_dict else None,
    )
    model = PatchedPredictorWrapper(model)
    model = model.eval()

    # ---- 3. Return ----
    return {
        "model": model,
        "name": model_info["name"],
        "data_module": data_module,
        "ckpt_file": model_info["ckpt_file"],
        "model_input_shape": model_input_shape,
        "model_output_shape": model_output_shape,
        "pad": pad_dict,
        "single_channel": model_info["single_channel"],
        "n_classes": n_classes,
        "batch_size": batch_size,
    }


def perform_inference(
    model_instantiator_func,
    batch_size=1,
    n_classes=6,
    img_shape: Tuple[int, int] = (1006, 590),
    accelerator: str = "gpu",
    devices: int = 1,
):
    model_info = load_model_and_data_module(
        model_instantiator_func=model_instantiator_func,
        img_shape=img_shape,
        n_classes=n_classes,
        batch_size=batch_size,
    )
    predictions_file = predictions_path / f"{model_info['name']}.npy"
    if predictions_file.exists():
        print(
            f"Predictions already exist at {predictions_file}. Skipping inference."
        )
        return None

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        max_epochs=1,
        enable_checkpointing=False,
    )
    predictions = trainer.predict(
        model_info["model"], model_info["data_module"]
    )
    predictions = torch.stack(predictions, dim=0) # type: ignore
    predictions = predictions.squeeze()
    predictions = predictions.float().cpu().numpy()
    np.save(predictions_file, predictions)
    
    print(f"Predictions saved at {predictions_file}. Shape: {predictions.shape}")
    return predictions_file
    


def main():
    predictions_path.mkdir(parents=True, exist_ok=True)

    for model_instantiator_func in [
        byol,
        fastsiam,
        kenshodense,
        simclr,
        tribyol,
        sam,
        lfr,
        dinov2_dpt,
        dinov2_mla,
        dinov2_pup,
        sfm_base_patch16,
    ]:
        model_name = model_instantiator_func.__name__
        print("-" * 80)
        
        try:
            print("*" * 20)
            print(f"Model: {model_name}")
            print("*" * 20)
            perform_inference(model_instantiator_func)
        except Exception as e:
            traceback.print_exc()
            print(f"Error executing model: {model_name}")
        print("-" * 80, "\n")


if __name__ == "__main__":
    main()
