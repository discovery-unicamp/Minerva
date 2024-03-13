import torch

from sslt.models.nets.setr import SETR_PUP

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SETR_PUP(
        image_size=(16, 32),
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=1,
        num_classes=3,
        num_convs=4,
        decoder_channels=256,
        up_scale=4,
        encoder_dropout=0.1,
        kernel_size=3,
        decoder_dropout=0.1,
        conv_act=torch.nn.ReLU(inplace=True),
    )
    model.to(device)
    result = model.forward(torch.zeros(1, 3, 16, 32).to(device))
    print(result[0].shape)
