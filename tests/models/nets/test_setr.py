import torch

from sslt.models.nets.setr import _SetR_PUP

if __name__ == "__main__":
    model = _SetR_PUP(
        image_size=512,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=2,
        mlp_dim=1,
        num_classes=3,
    )
    print(model)
    result = model.forward(torch.zeros(1, 2, 3, 4))
    print(result.shape)
    print(result)
