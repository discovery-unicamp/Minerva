from functools import partial
import os
from pathlib import Path
from typing import Optional, Tuple
import lightning as L
from torch.utils.data import DataLoader

from minerva.data.datasets.supervised_dataset import (
    SupervisedReconstructionDataset,
)
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.transforms.transform import _Transform

from typing import List, Union

from minerva.transforms.transform import (
    TransformPipeline,
    PadCrop,
    Identity,
    Transpose
)


def default_train_transforms(
    img_size: Tuple[int, int] = (1006, 590), seed: Optional[int] = None
) -> List[_Transform]:
    return [
        TransformPipeline(                      # Image reader transform
            [
                Transpose([2, 0, 1]),
                PadCrop(*img_size, padding_mode="reflect", seed=seed)
            ]
        ),
        TransformPipeline(                      # Label reader transform
            [
                PadCrop(*img_size, padding_mode="reflect", seed=seed),
            ]
        ),
    ]


def default_test_transforms() -> List[_Transform]:
    return [
        TransformPipeline([Transpose([2, 0, 1]),]),        # Image reader transform
        TransformPipeline([Identity()]),        # Label reader transform
    ]


TiffReaderWithNumericSort = partial(
    TiffReader, sort_method=["text", "numeric"], delimiter="_", key_index=[0, 1]
)
PNGReaderWithNumericSort = partial(
    PNGReader, sort_method=["text", "numeric"], delimiter="_", key_index=[0, 1]
)


class ParihakaDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_data_dir: str,
        root_annotation_dir: str,
        train_transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        test_transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.root_annotation_dir = Path(root_annotation_dir)
        self.train_transforms = train_transforms or default_train_transforms()
        self.test_transforms = test_transforms or default_test_transforms()
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.num_workers = num_workers or 1
        self.datasets = {}

    def setup(self, stage=None):
        if stage == "fit":
            train_img_reader = TiffReaderWithNumericSort(
                self.root_data_dir / "train"
            )
            train_label_reader = PNGReaderWithNumericSort(
                self.root_annotation_dir / "train",
            )
            train_dataset = SupervisedReconstructionDataset(
                readers=[train_img_reader, train_label_reader],
                transforms=self.train_transforms,
            )

            val_img_reader = TiffReaderWithNumericSort(
                self.root_data_dir / "val",
            )
            val_label_reader = PNGReaderWithNumericSort(
                self.root_annotation_dir / "val",
            )
            val_dataset = SupervisedReconstructionDataset(
                readers=[val_img_reader, val_label_reader],
                transforms=self.train_transforms,
            )

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            test_img_reader = TiffReaderWithNumericSort(
                self.root_data_dir / "test",
            )
            test_label_reader = PNGReaderWithNumericSort(
                self.root_annotation_dir / "test",
            )
            test_dataset = SupervisedReconstructionDataset(
                readers=[test_img_reader, test_label_reader],
                transforms=self.test_transforms,
            )
            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _get_dataloader(self, partition: str, shuffle: bool):
        return DataLoader(
            self.datasets[partition],
            batch_size=self.batch_size,
            num_workers=self.num_workers,  # type: ignore
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._get_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader("val", shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader("test", shuffle=False)

    def predict_dataloader(self):
        return self._get_dataloader("predict", shuffle=False)

    def __str__(self) -> str:
        return f"""DataModule
    Data: {self.root_data_dir}
    Annotations: {self.root_annotation_dir}
    Batch size: {self.batch_size}"""

    def __repr__(self) -> str:
        return str(self)
