from minerva.data.datasets import SimpleDataset
from minerva.data.readers import TiffReader, PNGReader
from minerva.transforms import TransformPipeline
from minerva.utils.typing import PathLike
from torchvision.transforms import Resize, ToTensor
from typing import Literal, Optional, Tuple
import torch

from pathlib import Path


class SeismicImageDataset(SimpleDataset):

    def __init__(
        self,
        root_dir: PathLike,
        subset: Literal["train", "val", "test"],
        resize: Optional[Tuple[int, int]] = None,
        labels: bool = True,
    ):
        """
        Seismic segmentation dataset in the form of tiff files, optionally annotated
        with single-channel pngs.
        
        Parameters
        ----------
        root_dir: PathLike
            The root directory where the dataset files are located. Directory structure
            must be
            ```
            root_dir
            ├── images
            │   ├── train
            │   │   └── file_0.tiff
            │   ├── val
            │   │   └── file_1.tiff
            │   └── test
            │       └── file_2.tiff
            └── annotations
                ├── train
                │   └── file_0.png
                ├── val
                │   └── file_1.png
                └── test
                    └── file_2.png
            ```
            where the annotation directory is optional.
        
        subset: Literal["train", "val", "test"]
            Which subset of the dataset to read from
        
        resize: Tuple[int, int], optional
            A shape to which to resize the images after reading them. If the dataset
            contains images of different shapes (e.g. inlines and crosslines) this is
            mandatory. If left as `None`, no resizing takes place. Defaults to `None`
        
        labels: bool
            Whether to return the segmentation annotation along with the seismic image.
            Must be `False` if the dataset does not contain annotations. Defaults to
            `True`
        """
        root_dir = Path(root_dir)
        readers = [TiffReader(root_dir / "images" / subset)]

        if resize:
            transforms = [
                TransformPipeline([ToTensor(), Resize(resize), lambda x: x.float()])
            ]
        else:
            transforms = [TransformPipeline([ToTensor(), lambda x: x.float()])]

        if labels:
            readers.append(PNGReader(root_dir / "annotations" / subset))
            if resize:
                transforms.append(
                    TransformPipeline(
                        [
                            lambda x: torch.from_numpy(x).unsqueeze(0),
                            Resize(resize),
                            lambda x: torch.squeeze(x, 0).long(),
                        ]
                    )
                )
            else:
                transforms.append(lambda x: torch.from_numpy(x).long())

        super().__init__(readers, transforms, not labels)
