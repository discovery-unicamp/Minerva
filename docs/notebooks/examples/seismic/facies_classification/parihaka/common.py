import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import lightning as L
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex

from minerva.data.datasets.supervised_dataset import (
    SupervisedReconstructionDataset,
)
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.tiff_reader import TiffReader
from minerva.models.loaders import FromPretrained
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.transforms.transform import _Transform, TransformPipeline

from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from minerva.data.readers.reader import _Reader


import tqdm


class PadCrop(_Transform):
    """Transforms image and pads or crops it to the target size.
    If the axis is larger than the target size, it will crop the image.
    If the axis is smaller than the target size, it will pad the image.
    """

    def __init__(
        self,
        target_h_size: int,
        target_w_size: int,
        padding_mode: str = "reflect",
        seed: int | None = None,
        constant_values: int = 0,
    ):
        """
        Initializes the transformation with target sizes, padding mode, and RNG seed.

        Parameters:
        - target_h_size (int): The target height size.
        - target_w_size (int): The target width size.
        - padding_mode (str): The padding mode to use (default is "reflect").
        - seed (int): Seed for random number generator to make cropping reproducible.
        """
        self.target_h_size = target_h_size
        self.target_w_size = target_w_size
        self.padding_mode = padding_mode
        self.rng = np.random.default_rng(
            seed
        )  # Random number generator with the provided seed
        self.constant_values = constant_values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        # print(f"-> [{self.__class__.__name__}] x.shape={x.shape}")

        # Handle height dimension independently: pad if target_h_size > h, else crop
        if self.target_h_size > h:
            pad_h = self.target_h_size - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_args = {
                "array": x,
                "pad_width": (
                    ((pad_top, pad_bottom), (0, 0), (0, 0))
                    if len(x.shape) == 3
                    else ((pad_top, pad_bottom), (0, 0))
                ),
                "mode": self.padding_mode,
            }
            if self.padding_mode == "constant":
                pad_args["constant_values"] = self.constant_values

            x = np.pad(**pad_args)

        elif self.target_h_size < h:
            crop_h_start = self.rng.integers(0, h - self.target_h_size + 1)
            x = x[crop_h_start : crop_h_start + self.target_h_size, ...]

        # Handle width dimension independently: pad if target_w_size > w, else crop
        if self.target_w_size > w:
            pad_w = self.target_w_size - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            pad_args = {
                "array": x,
                "pad_width": (
                    ((0, 0), (pad_left, pad_right), (0, 0))
                    if len(x.shape) == 3
                    else ((0, 0), (pad_left, pad_right))
                ),
                "mode": self.padding_mode,
            }

            if self.padding_mode == "constant":
                pad_args["constant_values"] = self.constant_values

            x = np.pad(**pad_args)

        elif self.target_w_size < w:
            crop_w_start = self.rng.integers(0, w - self.target_w_size + 1)
            x = x[:, crop_w_start : crop_w_start + self.target_w_size, ...]

        # Ensure channel dimension consistency
        if len(x.shape) == 2:  # For grayscale, add a channel dimension
            x = np.expand_dims(x, axis=2)

        # Convert to torch tensor with format C x H x W
        # output = torch.from_numpy(x).float()
        x = np.transpose(x, (2, 0, 1))  # Convert to C x H x W format
        # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
        # print(f"<- [{self.__class__.__name__}] x.shape={x.shape}")
        return x


class SelectChannel(_Transform):
    """Perform a channel selection on the input image."""

    def __init__(self, channel: int, expand_channels: int = None):
        """
        Initializes the transformation with the channel to select.

        Parameters:
        - channel (int): The channel to select.
        """
        self.channel = channel
        self.expand_channels = expand_channels

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x[self.channel, ...]
        if self.expand_channels is not None:
            x = np.expand_dims(x, axis=self.expand_channels)
        # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
        return x


class CastTo(_Transform):
    def __init__(self, dtype: type):
        """
        Initializes the transformation with the target data type.

        Parameters:
        - dtype (type): The target data type.
        """
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
        return x.astype(self.dtype)


class SwapAxes(_Transform):
    def __init__(self, source_axis: int, target_axis: int):
        """
        Initializes the transformation with the source and target axes.

        Parameters:
        - source_axis (int): The source axis to swap.
        - target_axis (int): The target axis to swap.
        """
        self.source_axis = source_axis
        self.target_axis = target_axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.swapaxes(x, self.source_axis, self.target_axis)
        # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
        return x


class RepeatChannel(_Transform):
    def __init__(self, repeats: int, axis: int):
        """
        Initializes the transformation with the number of repeats.

        Parameters:
        - repeats (int): The number of repeats.
        - axis (int): The axis to repeat.
        """
        self.repeats = repeats
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.repeat(x, self.repeats, axis=self.axis)
        # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
        return x


class ExpandDims(_Transform):
    def __init__(self, axis: int):
        """
        Initializes the transformation with the axis to expand.

        Parameters:
        - axis (int): The axis to expand.
        """
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.expand_dims(x, axis=self.axis)
        # print(f"[{self.__class__.__name__}] x.shape={x.shape}")
        return x


class PerSampleTransformPipeline(TransformPipeline):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        new_values = []
        for i, value in enumerate(x):
            # print(f"-->[{self.__class__.__name__}] value[{i}].shape={value.shape}")
            value = super().__call__(value)
            # print(f"<--[{self.__class__.__name__}] value[{i}].shape={value.shape}")
            new_values.append(value)

        return np.array(new_values)


def plot_image(img, title=None, cmap: str = "gray"):
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.show()


def plot_images(
    images,
    plot_title=None,
    subplot_titles=None,
    cmaps=None,
    filename=None,
    x_label=None,
    y_label=None,
    height=5,
    width=5,
    show=False,
):
    num_images = len(images)

    # Create a figure with subplots (1 row, num_images columns), adjusting size based on height and width parameters
    fig, axs = plt.subplots(1, num_images, figsize=(width * num_images, height))

    # Set overall plot title if provided
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16)

    # Ensure subplot_titles and cmaps are lists with correct lengths
    if subplot_titles is None:
        subplot_titles = [None] * num_images
    if cmaps is None:
        cmaps = ["gray"] * num_images

    # Plot each image in its respective subplot
    for i, (img, ax, title, cmap) in enumerate(
        zip(images, axs, subplot_titles, cmaps)
    ):
        im = ax.imshow(img, cmap=cmap)

        # Set title for each subplot if provided
        if title is not None:
            ax.set_title(title)

        # Add a colorbar for each subplot
        fig.colorbar(im, ax=ax)

        # Set x and y labels if provided
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

    # Adjust layout to fit titles, labels, and colorbars
    plt.tight_layout()

    # Save the figure if filename is provided
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Figure saved as '{filename}'")

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()


class MultiReader(_Reader):
    def __init__(self, readers):
        self.readers = readers

    def __len__(self) -> int:
        return len(self.readers[0])

    def __getitem__(self, i) -> np.ndarray:
        r = np.stack([reader[i] for reader in self.readers])
        # print(f"The shape of the reader is {r.shape}")
        return r


class GenericDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_data_dir: str,
        root_annotation_dir: str,
        transforms,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.root_annotation_dir = Path(root_annotation_dir)
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        self.datasets = {}

    def setup(self, stage=None):
        if stage == "fit":
            train_img_reader = TiffReader(self.root_data_dir / "train")
            train_label_reader = PNGReader(self.root_annotation_dir / "train")
            train_dataset = SupervisedReconstructionDataset(
                readers=[
                    train_img_reader,
                    train_label_reader,
                ],
                transforms=self.transforms,
            )

            val_img_reader = TiffReader(self.root_data_dir / "val")
            val_label_reader = PNGReader(self.root_annotation_dir / "val")
            val_dataset = SupervisedReconstructionDataset(
                readers=[
                    val_img_reader,
                    val_label_reader,
                ],
                transforms=self.transforms,
            )

            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        elif stage == "test" or stage == "predict":
            test_img_reader = TiffReader(self.root_data_dir / "test")
            test_label_reader = PNGReader(self.root_annotation_dir / "test")
            test_dataset = SupervisedReconstructionDataset(
                readers=[
                    test_img_reader,
                    test_label_reader,
                ],
                transforms=self.transforms,
            )
            self.datasets["test"] = test_dataset
            self.datasets["predict"] = test_dataset

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _get_dataloader(self, partition: str, shuffle: bool):
        return DataLoader(
            self.datasets[partition],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
