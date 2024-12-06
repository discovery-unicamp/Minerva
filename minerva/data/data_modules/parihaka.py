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
    Transpose,
    CastTo,
)


def default_train_transforms(
    img_size: Tuple[int, int] = (1006, 590), seed: int = 0
) -> List[_Transform]:
    """Default training transforms for the Parihaka dataset.
    Include a transform pipeline for the image reader and label reader.
    The image reader transform includes transposing the image to CxHxW format
    and padding/cropping the image to the specified size. The label reader
    transform only includes padding/cropping the label to the specified size.

    Parameters
    ----------
    img_size : Tuple[int, int], optional
        Parihaka inlines are (1006, 590) and crosslines are (1006, 531). We
        use the inline size as the default size for the image (1006, 590).
        By default (1006, 590)
    seed : Optional[int], optional
        Default seed for the random number generator. By default 0. The seed
        is the same for both the image and label reader transforms.

    Returns
    -------
    List[_Transform]
        A 2-element list of transform pipelines for the image and label reader
        transform pipelines.
    """
    return [
        TransformPipeline(  # Image reader transform
            [
                Transpose([2, 0, 1]),
                PadCrop(*img_size, padding_mode="reflect", seed=seed),
                CastTo("float32"),
            ]
        ),
        TransformPipeline(  # Label reader transform
            [
                PadCrop(*img_size, padding_mode="reflect", seed=seed),
                CastTo("int32"),
            ]
        ),
    ]


def default_test_transforms() -> List[_Transform]:
    """Default testing transforms for the Parihaka dataset. Testing transforms
    only include transposing the image to CxHxW format. No padding/cropping is
    applied to the image or label. Nothing is done to label.

    Returns
    -------
    List[_Transform]
        A 2-element list of transform pipelines for the image and label reader
        transform pipelines.
    """

    return [
        TransformPipeline(
            [Transpose([2, 0, 1]), CastTo("float32")]
        ),  # Image reader transform
        TransformPipeline([CastTo("int32")]),  # Label reader transform
    ]


# Partial functions for the TiffReader and PNGReader with numeric sort
# and delimiter "_" for the Parihaka dataset.
TiffReaderWithNumericSort = partial(
    TiffReader, sort_method=["text", "numeric"], delimiter="_", key_index=[0, 1]
)
PNGReaderWithNumericSort = partial(
    PNGReader, sort_method=["text", "numeric"], delimiter="_", key_index=[0, 1]
)


class ParihakaDataModule(L.LightningDataModule):
    """Default data module for the Parihaka dataset. This data module creates a
    supervised reconstruction dataset for training, validation, testing, and
    prediction with default transforms to read the images and labels.

    The parihaka dataset is organized as follows:
    root_data_dir
    ├── train
    │   ├── il_1.tif
    |   ├── il_2.tif
    |   ├── ...
    ├── val
    │   ├── il_1.tif
    |   ├── il_2.tif
    |   ├── ...
    ├── test
    │   ├── il_1.tif
    |   ├── il_2.tif
    |   ├── ...
    root_annotation_dir
    ├── train
    │   ├── il_1.png
    |   ├── il_2.png
    |   ├── ...
    ├── val
    │   ├── il_1.png
    |   ├── il_2.png
    |   ├── ...
    ├── test
    │   ├── il_1.png
    |   ├── il_2.png
    |   ├── ...

    The `root_data_dir` contains the seismic images and the
    `root_annotation_dir` contains the corresponding labels. Files with the
    same name in the same directory are assumed to be pairs of seismic images
    and labels. For instance `root_data_dir/train/il_1.tif` and
    `root_annotation_dir/train/il_1.png` are assumed to be a pair of seismic
    image and label.

    Original parihaka dataset contains inlines and crosslines in train and val
    directories. Inlines have dimensions (1006, 590, 3) and crosslines have
    dimensions (1006, 531, 3). By default, crosslines are padded to (1006, 590)
    and all images are transposed to (3, 1006, 590) format. Labels are also
    padded to (1006, 590) and are not transposed. Finally, images are cast to
    float32 and labels are cast to int32.
    """

    def __init__(
        self,
        root_data_dir: str,
        root_annotation_dir: str,
        train_transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        valid_transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        test_transforms: Optional[Union[_Transform, List[_Transform]]] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        drop_last: bool = True,
    ):
        """Initialize the ParihakaDataModule with the root data and annotation
        directories. The data module is initialized with default training and
        testing transforms.

        Parameters
        ----------
        root_data_dir : str
            Root directory containing the seismic images. Inside this directory
            should be subdirectories `train`, `val`, and `test` containing the
            training, validation, and testing TIFF images.
        root_annotation_dir : str
            Root directory containing the annotations. Inside this directory
            should be subdirectories `train`, `val`, and `test` containing the
            training, validation, and testing PNG annotations. Files with the
            same name in the same directory are assumed to be pairs of seismic
            images and labels.
        train_transforms : Optional[Union[_Transform, List[_Transform]]], optional
            2-element list of transform pipelines for the image and label reader.
            Transforms to apply to the training and validation datasets. If
            None, default training transforms are used, which pads images to
            (1006, 590) and transposes them to (3, 1006, 590) format. Labels
            are also padded to (1006, 590). By default None
        valid_transforms: Optional[Union[_Transform, List[_Transform]]], optional
            2-element list of transform pipelines for the image and label reader.
            Transforms to apply to the validation datasets. If None, default
            training transforms are used, which pads images to (1006, 590) and
            transposes them to (3, 1006, 590) format. Labels are also padded to
            (1006, 590). By default None
        test_transforms : Optional[Union[_Transform, List[_Transform]]], optional
            2-element list of transform pipelines for the image and label reader.
            Transforms to apply to the testing and prediction datasets. If None,
            default testing transforms are used, which transposes images to
            CxHxW format. Labels are untouched. By default None
        batch_size : int, optional
            Default batch size for the dataloaders, by default 1
        num_workers : Optional[int], optional
            Number of workers for the dataloaders, by default None. If None,
            the number of workers is set to the number of CPUs on the system.
        drop_last : bool, optional
            Whether to drop the last batch if it is smaller than the batch size,
            by default True.
        """
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.root_annotation_dir = Path(root_annotation_dir)
        self.train_transforms = train_transforms or default_train_transforms()
        self.valid_transforms = valid_transforms or default_train_transforms()
        self.test_transforms = test_transforms or default_test_transforms()
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.num_workers = num_workers or 1
        self.drop_last = drop_last
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
                transforms=self.valid_transforms,
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
            drop_last=self.drop_last,
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

