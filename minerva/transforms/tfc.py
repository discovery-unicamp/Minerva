import numpy as np
import torch
import torch.fft as fft
from torch import nn
from .transform import _Transform
from typing import Union, Tuple


class TFC_Transforms(_Transform):
    """
    Transformations used in the TFC model.
    It consists of time and frequency domain data augmentation.
    """

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method that applies the transformations to the input data.

        Parameters
        ----------
        - x: Union[np.ndarray, torch.Tensor]
            The input data to be transformed

        Returns
        -------
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple with the original data, the transformed data in the time domain the frequency version of the data and the tranformed data in frequency domain
        """

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.FloatTensor)
            device = torch.device("cpu")
        elif isinstance(x, torch.Tensor):
            device = x.device
            x = x.type(torch.FloatTensor)
        else:
            print("The type of the input is: ", type(x), "It is ", x)
            raise TypeError("The input data must be a numpy array or a torch tensor")
        freq = fft.fft(x).abs()
        y1 = self.DataTransform_TD(x)
        y2 = self.DataTransform_FD(freq)
        return (
            x.type(torch.FloatTensor).to(device),
            y1.type(torch.FloatTensor).to(device),
            freq.type(torch.FloatTensor).to(device),
            y2.type(torch.FloatTensor).to(device),
        )

    def one_hot_encoding(self, X: np.ndarray, n_values: int = None):
        """
        One-hot encoding of the input data

        Parameters
        ----------
        - X: np.ndarray
            The input data to be encoded
        - n_values: int
            The number of classes in the data. If None, the number of classes is inferred from the data

        Returns
        -------
        - np.ndarray
            The one-hot encoded data
        """
        X = [int(x) for x in X]
        if n_values is None:
            n_values = np.max(X) + 1
        b = torch.eye(n_values)[X]
        return b

    def DataTransform_TD(
        self, sample: np.ndarray, jitter_ratio: float = 0.8
    ) -> np.ndarray:
        """
        Weak and strong augmentations.
        Consists of jittering and removing time components.

        Parameters
        ----------
        - sample: np.ndarray
            The input data to be augmented
        - jitter_ratio: float
            The ratio of the jittering transformation

        Returns
        -------
        - np.ndarray
            The augmented data

        """
        aug_1 = self.jitter(sample, jitter_ratio)

        li = torch.randint(0, 4, (sample.shape[0],))
        li_onehot = self.one_hot_encoding(li)
        aug_1[(1 - li_onehot[:, 0]).bool()] = 0

        return aug_1

    def DataTransform_FD(self, sample: np.ndarray) -> np.ndarray:
        """
        Weak and strong augmentations.
        Consists of jittering and adding or removing frequency components.

        Parameters
        ----------
        - sample: np.ndarray
            The input data to be augmented

        Returns
        -------
        - np.ndarray
            The augmented data

        """
        aug_1 = self.remove_frequency(sample, 0.1)
        aug_2 = self.add_frequency(sample, 0.1)

        li = torch.randint(0, 2, (sample.shape[0],))
        li_onehot = self.one_hot_encoding(li, 2)
        aug_1[(1 - li_onehot[:, 0]).bool()] = 0
        aug_2[(1 - li_onehot[:, 1]).bool()] = 0
        aug_F = aug_1 + aug_2
        return aug_F

    def jitter(self, x: np.ndarray, sigma: float = 0.8):
        """
        Add noise to the input data.

        Parameters
        ----------
        - x: np.ndarray
            The input data to be augmented
        - sigma: float
            The standard deviation of the noise

        Returns
        -------
        - np.ndarray
            The data with added noise


        """
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + torch.normal(0.0, sigma, x.shape)

    def remove_frequency(self, x: np.ndarray, maskout_ratio: float = 0):
        """
        function to remove frequency components from the input data.

        Parameters
        ----------
        - x: np.ndarray
            The input data to be augmented
        - maskout_ratio: float
            The ratio of the frequency components to be removed

        Returns
        -------
        - np.ndarray
            The data with removed frequency components
        """
        # verify if on gpu
        if x.device == torch.device("cpu"):
            mask = torch.FloatTensor(x.shape).uniform_() > maskout_ratio
        else:
            mask = (
                torch.cuda.FloatTensor(x.shape).uniform_() > maskout_ratio
            )  # maskout_ratio are False
        mask = mask.to(x.device)
        return x * mask

    def add_frequency(
        self,
        x: np.ndarray,
        pertub_ratio: float = 0,
    ):
        """
        function to add frequency components to the input data.

        Parameters
        ----------
        - x: np.ndarray
            The input data to be augmented
        - pertub_ratio: float
            The ratio of the frequency components to be added

        Returns
        -------
        - np.ndarray
            The data with added frequency components


        """
        if x.device == torch.device("cpu"):
            mask = torch.FloatTensor(x.shape).uniform_() > (1 - pertub_ratio)
        else:
            mask = torch.cuda.FloatTensor(x.shape).uniform_() > (
                1 - pertub_ratio
            )  # only pertub_ratio of all values are True
        mask = mask.to(x.device)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape) * (max_amplitude * 0.1)
        pertub_matrix = mask * random_am
        return x + pertub_matrix
