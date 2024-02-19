from typing import Tuple
import torch


import lightning as L
import torch


from torch.utils.data import DataLoader

from typing import List, Tuple, Dict
import numpy as np
import torch


from torchmetrics.image import (
    PeakSignalNoiseRatio,
)


import lightning as L
import torch




def get_split_data_loader(stage: str, data_module: L.LightningDataModule) -> DataLoader:
    if stage == "train":
        data_module.setup("fit")
        return data_module.train_dataloader()
    elif stage == "validation":
        data_module.setup("fit")
        return data_module.val_dataloader()
    elif stage == "test":
        data_module.setup("test")
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Invalid stage: {stage}")

def dataset_from_dataloader(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y = [], []
    for batch in dataloader:
        X.append(batch[0])
        y.append(batch[1])
    return torch.cat(X), torch.cat(y)


def compute_reconstruction_metrics(y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    metrics = {}
    metrics["mse"] = float(torch.mean((y - y_hat) ** 2).item())
    metrics["mae"] = float(torch.mean(torch.abs(y - y_hat)).item())
    metrics["psnr"] = float(PeakSignalNoiseRatio()(y_hat, y).item())
    return metrics

def reconstruct_from_patches(patches: List[torch.Tensor], coords: List[Tuple[int, ...]]) -> np.ndarray:
    """
    Reconstructs an image from a list of patches and their corresponding coordinates.

    Args:
    - patches (List[torch.Tensor]): A list of tensors where each tensor is a patch.
    - coords (List[Tuple[int, ...]]): A list of tuples where each tuple represents the coordinates
        of the top-left corner of each patch in the final image.

    Returns:
    - np.ndarray: The reconstructed image.
    """
    # Determine the maximum shape among all patches
    max_shape = np.max([patch.shape for patch in patches], axis=0)

    # Calculate the shape of the final matrix based on the maximum coordinates and maximum shape
    final_shape = tuple(np.max(coords, axis=0) + max_shape)

    # Initialize the final matrix with zeros
    final_matrix = np.zeros(
        final_shape, dtype=np.float32
    )  # Assuming patches is not empty

    # Place each patch from list patches into the final matrix at the specified coordinates
    for coord, patch in zip(coords, patches):
        slices = tuple(
            slice(coord[i], coord[i] + patch.shape[i])
            for i in range(len(coord))
        )
        final_matrix[slices] = patch.numpy().astype(np.float32)

    return final_matrix
