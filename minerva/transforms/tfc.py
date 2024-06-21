import numpy as np
import torch
import torch.fft as fft
from torch import nn
from .transform import _Transform

class TFC_Transforms(_Transform):
    def __call__(self, x):
        device = x.device
        tipo = type(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.FloatTensor)
        else:
            x = x.type(torch.FloatTensor)
        freq = fft.fft(x).abs()
        y1 = self.DataTransform_TD(x)
        y2 = self.DataTransform_FD(freq)
        x, y1, freq, y2 = x.type(tipo), y1.type(tipo), freq.type(tipo), y2.type(tipo)
        x, y1, freq, y2 = x.to(device), y1.to(device), freq.to(device), y2.to(device)
        return x, y1, freq, y2


    def one_hot_encoding(self, X, n_values=None):
        X = [int(x) for x in X]
        if n_values is None:
            n_values = np.max(X) + 1
        b = np.eye(n_values)[X]
        return b

    def DataTransform_TD(self, sample, jitter_ratio = 0.8):
        """Weak and strong augmentations"""
        aug_1 = self.jitter(sample, jitter_ratio)

        li = np.random.randint(0, 4, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = self.one_hot_encoding(li)
        aug_1[1-li_onehot[:, 0]] = 0 # the rows are not selected are set as zero.
        return aug_1


    def DataTransform_FD(self, sample):
        """Weak and strong augmentations in Frequency domain """
        aug_1 =  self.remove_frequency(sample, 0.1)
        aug_2 = self.add_frequency(sample, 0.1)
        li = np.random.randint(0, 2, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = self.one_hot_encoding(li,2)
        aug_1[1-li_onehot[:, 0]] = 0 # the rows are not selected are set as zero.
        aug_2[1 - li_onehot[:, 1]] = 0
        aug_F = aug_1 + aug_2
        return aug_F


    def jitter(self, x, sigma=0.8):
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

    def remove_frequency(self, x, maskout_ratio=0):
        mask = torch.cuda.FloatTensor(x.shape).uniform_() > maskout_ratio # maskout_ratio are False
        mask = mask.to(x.device)
        return x*mask
        

    def add_frequency(self, x, pertub_ratio=0,):
        mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
        mask = mask.to(x.device)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
        pertub_matrix = mask*random_am
        return x+pertub_matrix