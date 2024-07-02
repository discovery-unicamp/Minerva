from typing import Any
import numpy as np
from scipy import signal

class ActivityImageTransforms:
    def __init__(self, signal_sequences=9, signal_samples=68) -> None:
        self.signal_sequences = signal_sequences
        self.signal_samples = signal_samples
    
    def get_signal_image(self, raw_signals):
        i = 1
        j = i + 1
        signal_image = [raw_signals[i-1]]
        signal_index_string = str(i)
        
        while i != j:
            keys1 = str(i)+str(j)
            keys2 = str(j)+str(i)

            if j > self.signal_sequences:
                j = 1
            elif  (not keys1 in signal_index_string) and (not keys2 in signal_index_string):
                signal_image.append(raw_signals[j-1])
                signal_index_string += str(j)
                i = j
                j += 1  
            else:
                j += 1
        return signal_image
    
    def resample_signals(self, x):
        resample_signal = []
        for raw_signal in x:
            resample_signal.append(signal.resample(raw_signal, self.signal_samples))
        return resample_signal    
    
    def __call__(self, x) -> Any:
        raw_signals = self.resample_signals(x)
        signal_image = np.array(self.get_signal_image(raw_signals))
        dft_image = np.fft.fft2(signal_image, axes=(0, 1))
        dft_image_shifted = np.fft.fftshift(dft_image)
        activity_image = np.abs(dft_image_shifted)    
        return x, activity_image
