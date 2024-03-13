import torch

from sslt.models.nets.setr import SETR_PUP

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SETR_PUP()
    model.to(device)
    result = model.forward(torch.zeros(1, 3, 512, 512).to(device))
    print(result[0].shape)
