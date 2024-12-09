import pytest
import torch
from torchinfo import summary

from minerva.models.nets.image.sam import Sam
from minerva.models.lora_adapters.simple_lora import LoRALinear

def test_sam_with_lora_loss():
    apply_freeze={"prompt_encoder": True, "image_encoder": True, "mask_decoder": True}
    apply_adapter={"image_encoder": LoRALinear, "mask_decoder": LoRALinear}

    model = Sam(
        apply_freeze=apply_freeze,
        apply_adapter=apply_adapter
    )

    batch = [{
        "image": torch.rand(3, 255, 701),
        "label": torch.rand(1, 255, 701).long(),
        "original_size": (255, 701),
        "multimask_output": True,
    }]

    summary_info = summary(model, verbose=0)
    trainable_params = summary_info.trainable_params

    # Do the training step
    loss = model.training_step(batch, 0).item()
    assert loss is not None
    assert loss >= 0, f"Expected non-negative loss, but got {loss}"

    # check trainable params
    assert trainable_params is not None
    assert trainable_params == 191056, f"Expected exactly 191056 trainable params, but got {trainable_params}"