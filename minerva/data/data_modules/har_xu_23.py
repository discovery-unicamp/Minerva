import os
from pathlib import Path
from typing import List
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics.pairwise import cosine_similarity
import torch

class TNCDataset(Dataset):
    def __init__(self, x, mc_sample_size, window_size, epsilon=3, adf=True):
        super(TNCDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1] 
        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.adf = adf
        if not self.adf:
            self.epsilon = epsilon
            self.delta = 5*window_size*epsilon

    def __len__(self):
        return self.time_series.shape[0]

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        print(f'ind: {ind}, window_size: {self.window_size}, {2*self.window_size},{self.T},{self.T-2*self.window_size}')
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        x_t = torch.from_numpy(self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]).to(torch.float).transpose(-1,-2)
        X_close = torch.from_numpy(self._find_neighours(self.time_series[ind], t)).to(torch.float).transpose(-1,-2)
        X_distant = torch.from_numpy(self._find_non_neighours(self.time_series[ind], t)).to(torch.float).transpose(-1,-2)


        return x_t, X_close, X_distant

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size,4*self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if np.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5*self.epsilon*self.window_size

            t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
            t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]
            x_p = np.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        else:

            target_window = x[:, t - self.window_size // 2:t + self.window_size // 2].flatten()
            similarities = []
            gap = self.window_size
            for w_t in range(self.window_size, T - self.window_size, gap):
                window = x[:, w_t - self.window_size // 2:w_t + self.window_size // 2].flatten()
                cos_sim = cosine_similarity([target_window], [window])[0][0]
                similarities.append((w_t, cos_sim))


            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            t_p = [w_t for w_t, _ in similarities[:self.mc_sample_size]]

            t_p = [max(self.window_size // 2 + 1, min(t_pp, T - self.window_size // 2)) for t_pp in t_p]
            x_p = np.stack([x[:, t_ind - self.window_size // 2:t_ind + self.window_size // 2] for t_ind in t_p])
        return x_p 


    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        x_n = np.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n
        
class HarDataModule(L.LightningDataModule):
    def __init__(
        self,
        processed_data_dir: str = "data/har/processed",
        batch_size: int = 16,
        mc_sample_size: int = 5,
        epsilon: int = 3,
        adf: bool = True,
        subseq_size: int = 128,
    ):
        super().__init__()
        self.processed_data_dir = Path(processed_data_dir)
        self.batch_size = batch_size
        self.mc_sample_size = mc_sample_size
        self.epsilon = epsilon
        self.adf = adf
        self.subseq_size = subseq_size

        self.setup()

    def setup(self):
        processedharpath = self.processed_data_dir

        self.har_train = np.load(os.path.join(processedharpath, "train_data.npy"))
        self.har_val = np.load(os.path.join(processedharpath, "val_data.npy"))
        self.har_test = np.load(os.path.join(processedharpath, "test_data.npy"))

    def train_dataloader(self):
        return DataLoader(TNCDataset(np.transpose(self.har_train, (0, 2, 1)), self.mc_sample_size, self.subseq_size, self.epsilon, self.adf), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(TNCDataset(np.transpose(self.har_val, (0, 2, 1)), self.mc_sample_size, self.subseq_size, self.epsilon, self.adf), batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(TNCDataset(np.transpose(self.har_test, (0, 2, 1)), self.mc_sample_size, self.subseq_size, self.epsilon, self.adf), batch_size=self.batch_size, shuffle=False)
