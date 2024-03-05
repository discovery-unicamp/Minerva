import torch

from sslt.models.nets.setr import _SetR_PUP

if __name__ == "__main__":
    model = _SetR_PUP(2, 3, 4, 5, 6, 7, 8, 9, 10)
    print(model)
    result = model.forward(torch.zeros(1, 2, 3, 4))
    print(result.shape)
    print(result)
