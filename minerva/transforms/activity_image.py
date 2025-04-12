from typing import Any
import numpy as np
from scipy import signal


class ActivityImageTransforms:
    """
    A class to transform signals into images for Human Activity Recognition (HAR),
    based on the methodology presented in the paper
    "Human Activity Recognition using Wearable Sensors by Deep Convolutional Neural Networks."
    """

    def __init__(self, signal_sequences=9, signal_samples=68) -> None:
        """Initializes the ActivityImageTransforms class with the given number of signal sequences and samples.

        Parameters:
        -----------
        signal_sequences: int, optional
            The number of signal sequences. Defaults to 9
        signal_samples: int, optional
            The number of samples per signal sequence. Defaults to 68.
        """
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

    def __call__(self, x) -> Any:
        """Transforms the input signals into activity images suitable for CNN processing.

        This method performs the following steps:
        1. Resamples the input signals to ensure consistent sample length.
        2. Constructs a 2D signal image from the resampled signals.
        3. Applies a 2D Fast Fourier Transform (FFT) to convert the signal image to the frequency domain.
        4. Shifts the zero-frequency component to the center of the spectrum.
        5. Computes the magnitude of the frequency domain representation to form the final activity image.

        Parameters:
        -----------
        x: (List[np.ndarray])
            The raw signal sequences from wearable sensors.

        Returns:
        --------
            tuple: A tuple containing the original signals and the corresponding activity images.
        """
        raw_signals = self.resample_signals(x)
        signal_image = np.array(self.get_signal_image(raw_signals))
        dft_image = np.fft.fft2(signal_image, axes=(0, 1))
        dft_image_shifted = np.fft.fftshift(dft_image)
        activity_image = np.abs(dft_image_shifted)
        return x, activity_image
