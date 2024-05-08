import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex

from minerva.data.datasets.supervised_dataset import SupervisedReconstructionDataset
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.models.nets.setr import SETR_PUP
from minerva.transforms.transform import Flip, TransformPipeline, _Transform


class Padding(_Transform):
    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)
        if len(x.shape) == 2:
            padded = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")
            padded = np.expand_dims(padded, axis=2)
        else:
            padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            padded = padded.astype(float)

        padded = np.transpose(padded, (2, 0, 1))
        return padded


transform = Padding(768)

train_att_reader = TiffReader("f3/images/train")
val_att_reader = TiffReader("f3/images/val")
test_att_reader = TiffReader("f3/images/test")

train_lbl_reader = PNGReader("f3/annotations/train")
val_lbl_reader = PNGReader("f3/annotations/val")
test_lbl_reader = PNGReader("f3/annotations/test")

train_dataset = SupervisedReconstructionDataset(
    [train_att_reader, train_lbl_reader], transform
)
val_dataset = SupervisedReconstructionDataset(
    [val_att_reader, val_lbl_reader], transform
)
test_dataset = SupervisedReconstructionDataset(
    [test_att_reader, test_lbl_reader], transform
)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class F3DataModule(L.LightningDataModule):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader):
        super().__init__()
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


data_module = F3DataModule(train_dataloader, val_dataloader, test_dataloader)

model = SETR_PUP(
    image_size=768,
    num_classes=6,
    log_train_metrics=True,
    train_metrics=JaccardIndex(task="multiclass", num_classes=6),
)


trainer = L.Trainer(max_epochs=200, fast_dev_run=4)
trainer.fit(model, data_module)
trainer.save_checkpoint("setr.pth")
