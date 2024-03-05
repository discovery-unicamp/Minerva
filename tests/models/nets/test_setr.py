import torch

from sslt.models.nets.setr import _SetR_PUP

if __name__ == "__main__":
    model = _SetR_PUP(512, 16, 24, 16, 1, 1, 3)
    print(model)
    result = model.forward(torch.zeros(1, 2, 3, 4))
    print(result.shape)
    print(result)
