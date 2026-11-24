from typing import Any, Dict, List

import numpy as np
from scipy import signal

from .transform import _as_keys, _Transform


class ActivityImageTransform(_Transform):
    """
    A class to transform signals into images for Human Activity Recognition (HAR),
    based on the methodology presented in the paper
    "Human Activity Recognition using Wearable Sensors by Deep Convolutional Neural Networks."
    """

    def __init__(
        self,
        keys_to_transform: List[str],
        signal_sequences: int = 9,
        signal_samples: int = 68,
    ) -> None:
        """Initializes the transform.

        For each listed key ``k`` the raw signals stay in ``k`` and the
        activity image is written to the new key ``{k}_activity_image``.

        Parameters:
        -----------
        keys_to_transform: List[str]
            Keys of the raw signal sequences to convert.
        signal_sequences: int, optional
            The number of signal sequences. Defaults to 9
        signal_samples: int, optional
            The number of samples per signal sequence. Defaults to 68.
        """
        self.keys_to_transform = _as_keys(keys_to_transform)
        self.signal_sequences = signal_sequences
        self.signal_samples = signal_samples

    def get_signal_image(self, raw_signals):
        """
        Constructs a signal image from the raw signals by arranging them in a specific order.

        This method follows the approach of combining sensor signals to form a 2D image, which is then
        suitable for CNN processing. The image is constructed by iteratively selecting signal sequences
        in a way that maximizes the spatial correlation between different sequences.

        Parameters:
        -----------
        raw_signals: np.ndarray
            The raw signal sequences from wearable sensors.

        Returns:
        --------
        signal_image: np.ndarray
            The constructed signal image, ready for further processing by a CNN.
        """
        i = 1
        j = i + 1
        signal_image = [raw_signals[i - 1]]
        signal_index_string = str(i)

        while i != j:
            keys1 = str(i) + str(j)
            keys2 = str(j) + str(i)

            if j > self.signal_sequences:
                j = 1
            elif (not keys1 in signal_index_string) and (
                not keys2 in signal_index_string
            ):
                signal_image.append(raw_signals[j - 1])
                signal_index_string += str(j)
                i = j
                j += 1
            else:
                j += 1
        return signal_image

    def resample_signals(self, x):
        """Resamples the signals to a fixed number of samples.

        This resampling step ensures that all signal sequences have the same length, which is necessary
        for constructing a consistent 2D image for CNN input.

        Parameters:
        -----------
        x: np.array
            The raw signal sequences from wearable sensors.

        Returns:
        --------
        resample_signal: List[np.ndarray]
            The resampled signals, all having the same number of samples.
        """
        resample_signal = []
        for raw_signal in x:
            resample_signal.append(signal.resample(raw_signal, self.signal_samples))
        return resample_signal

    def activity_image(self, x) -> np.ndarray:
        """Build the activity image of one set of raw signal sequences.

        Steps: resample the signals to a fixed length, arrange them into a 2D
        signal image, apply a 2D FFT, shift the zero frequency to the center
        and take the magnitude.

        Parameters:
        -----------
        x: (List[np.ndarray])
            The raw signal sequences from wearable sensors.

        Returns:
        --------
        np.ndarray
            The activity image.
        """
        raw_signals = self.resample_signals(x)
        signal_image = np.array(self.get_signal_image(raw_signals))
        dft_image = np.fft.fft2(signal_image, axes=(0, 1))
        return np.abs(np.fft.fftshift(dft_image))

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys_to_transform:
            data[f"{key}_activity_image"] = self.activity_image(data[key])
        return data
