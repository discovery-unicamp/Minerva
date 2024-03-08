import torch

from sslt.models.nets.setr import _SetR_PUP

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _SetR_PUP(
        image_size=512,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=1,
        num_classes=3,
    )
    model.to(device)
    result = model.forward(torch.zeros(1, 3, 512, 512).to(device))
    print(result[0].shape)
