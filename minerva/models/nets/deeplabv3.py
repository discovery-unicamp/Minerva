from torch import nn, Tensor, optim, load
from typing import Sequence

from torchvision.models.resnet import resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP
from minerva.models.nets.base import SimpleSupervisedModel

class DeepLabV3Model(SimpleSupervisedModel):
    def __init__(self, 
                 backbone: nn.Module=None, 
                 pred_head: nn.Module=None,
                 loss_fn: nn.Module=None,
                 learning_rate: float=0.001,
                 num_classes:int=6):
        backbone = backbone or DeepLabV3Backbone()
        pred_head = pred_head or DeepLabV3PredictionHead(num_classes=num_classes)
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        super().__init__(backbone=backbone,
                         fc=pred_head,
                         loss_fn=loss_fn,
                         learning_rate=learning_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        input_shape = x.shape[-2:]
        h = self.backbone(x)
        z = self.fc(h)
        # Upscaling
        return nn.functional.interpolate(z, size=input_shape, 
                                         mode="bilinear", align_corners=False)

    def _loss_func(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss_fn(y_hat, y.squeeze(1).long()) 

    def configure_optimizers(self):
        return optim.Adam(params=self.parameters(), lr=self.learning_rate)    

class DeepLabV3Backbone(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        RN50model = resnet50(replace_stride_with_dilation=[False, True, True])
        self.RN50model = RN50model
    
    def freeze_weights(self):
        for param in self.RN50model.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.RN50model.parameters():
            param.requires_grad = True

    def forward(self, x):
            x = self.RN50model.conv1(x)
            x = self.RN50model.bn1(x)
            x = self.RN50model.relu(x)
            x = self.RN50model.maxpool(x)
            x = self.RN50model.layer1(x)
            x = self.RN50model.layer2(x)
            x = self.RN50model.layer3(x)
            x = self.RN50model.layer4(x)
            #x = self.RN50model.avgpool(x)      # These should be removed for deeplabv3
            #x = torch.RN50model.flatten(x, 1)  # These should be removed for deeplabv3
            #x = self.RN50model.fc(x)           # These should be removed for deeplabv3
            return x
    
class DeepLabV3PredictionHead(nn.Sequential):
    def __init__(self, 
                 in_channels: int = 2048, 
                 num_classes: int = 6, 
                 atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )