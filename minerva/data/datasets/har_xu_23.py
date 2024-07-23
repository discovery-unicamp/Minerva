import numpy as np
from torch.utils.data import Dataset
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics.pairwise import cosine_similarity
import torch
from minerva.utils.typing import PathLike
from typing import List, Tuple
import os

class TNCDataset(Dataset):
    def __init__(
        self,
        x: np.array,
        mc_sample_size: int = 5,
        window_size: int = 128,
        epsilon=3,
        adf: bool = True,
    ):
        """
        This TNCDataset class is designed to handle time series data for the TNC 
        (Temporal Neighborhood Coding) task. It includes methods to load data, find close neighbors 
        using ADF testing or cosine similarity, and find distant non-neighbors.
        The dataset returns a tuple of the central window, close neighbors, and distant non-neighbors for each sample.

        Parameters
        ----------
        x : np.ndarray
            The time series data of shape (n_samples, n_channels, n_timesteps).
        mc_sample_size : int
            This value determines how many neighboring and non-neighboring windows are used per data sample.
        window_size : int
            The size of the window to be used for each sample.
        epsilon : int, optional
            This parameter controls the "spread" of neighboring windows. 
            Higher values lead to more diverse neighbors within a larger search radius around the center window.
        adf : bool, optional
            A flag indicating whether to use ADF (Augmented Dickey-Fuller) testing for finding neighbors. Defaults to True.
        
        Neighbor Selection
        ------------------
        The selection of neighbors and non-neighbors is crucial for TNC. Here's how it's done:

        1. **Finding Close Neighbors**:
            - **ADF (Augmented Dickey-Fuller) Testing**:
                - The ADF test checks the stationarity of the time series segments.
                - For each time window size `w_t` (ranging from `window_size` to `4 * window_size`), 
                the ADF test is applied to determine the p-value.
                - The average p-value across all channels is calculated.
                - The neighborhood size `epsilon` is determined based on the p-values. If all p-values 
                are below the threshold (0.01), `epsilon` is set to the length of `corr`, 
                otherwise, it is set to the first index where the p-value exceeds 0.01.
                - The `delta` is then set to `5 * epsilon * window_size`.
                - Neighboring time steps are generated by adding a random value from a normal distribution scaled
                by `epsilon * window_size` to the current time step `t`.
                - These time steps are adjusted to ensure they are within valid bounds.

            - **Cosine Similarity**:
                - If ADF is not used, cosine similarity is employed to find close neighbors.
                - The target window (current segment) is flattened, and its cosine similarity with all 
                other windows of the same size in the time series is calculated.
                - The top `mc_sample_size` windows with the highest cosine similarity are selected as neighbors.
                - The selected time steps are adjusted to ensure they are within valid bounds.

        2. **Finding Distant Non-Neighbors**:
            - The method `_find_non_neighbors` generates non-neighbors by selecting time steps far 
            from the current time step `t`.
            - Depending on whether `t` is in the first or second half of the time series, the non-neighbor
            time steps are selected to be either before or after the `delta` range.
            - A fallback mechanism ensures at least one non-neighbor segment is returned, 
            even if the primary selection fails.
             
        """
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
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return self.time_series.shape[0]

    def __getitem__(self, ind):
        """
        Returns a sample from the dataset.

        Parameters
        ----------
        ind : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the central window, close neighbors, and distant non-neighbors.
        """
        ind = ind%len(self.time_series)
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        x_t = torch.from_numpy(self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]).to(torch.float).transpose(-1,-2)
        X_close = torch.from_numpy(self._find_neighours(self.time_series[ind], t)).to(torch.float).transpose(-1,-2)
        X_distant = torch.from_numpy(self._find_non_neighours(self.time_series[ind], t)).to(torch.float).transpose(-1,-2)


        return x_t, X_close, X_distant

    def _find_neighours(self, x, t):
        """
        Finds close neighbors for a given time step.

        Parameters
        ----------
        x : np.ndarray
            The time series data for a single sample.
        t : int
            The current time step.

        Returns
        -------
        np.ndarray
            An array of close neighbors.
        """
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
        """
        Finds distant non-neighbors for a given time step.

        Parameters
        ----------
        x : np.ndarray
            The time series data for a single sample.
        t : int
            The current time step.

        Returns
        -------
        np.ndarray
            An array of distant non-neighbors.
        """
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
    
class HarDataset(Dataset):
    def __init__(
        self,
        data_path: PathLike,
        annotate: str,
        feature_column_prefixes: List[str] = [
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        target_column: str = "standard activity code",
        flatten: bool = False,
    ):
        """
        Dataset class for loading and preparing human activity recognition data.

        Parameters
        ----------
        data_path : PathLike
            Path to the directory containing the dataset files.
            It must have 6 files, named:
            train_data_subseq.npy, train_labels_subseq.npy,
            val_data.npy, val_labels_subseq.npy,
            test_data.npy, and test_labels_subseq.npy.
            This files corresponds to data segmented into subsequences of a fixed length (e.g., 128 samples). 
            These data subsequences are used for the downstream model, allowing it to learn patterns within these smaller segments.
            The labels are the labels for each subsequence in each set, going drom 0 to 5.
        annotate : str
            Annotation type for the dataset (e.g., 'train', 'val', 'test').
        feature_column_prefixes : List[str], optional
            Prefixes for the feature columns in the dataset. Defaults to accelerometer and gyroscope data.
        target_column : str, optional
            Column name for the target variable. Defaults to 'standard activity code'.
        flatten : bool, optional
            Whether to flatten the input data. Defaults to False.
        """
        super().__init__()
        self.data_path = data_path
        self.annotate = annotate
        self.feature_column_prefixes = feature_column_prefixes
        self.target_column = target_column
        self.flatten = flatten

        self.data = np.load(os.path.join(self.data_path, f"{self.annotate}_data_subseq.npy"))
        self.labels = np.load(os.path.join(self.data_path, f"{self.annotate}_labels_subseq.npy"))

        # self.labels = np.load(self.data_path / f"{self.annotate}_labels_subseq.npy")
        assert len(self.data) == len(self.labels), "Data and labels must have the same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Tuple containing the features and the target label.
        """
        data = self.data[idx]
        if self.flatten:
            data = data.flatten()

        features = data
        target = self.labels[idx]

        # Convert to torch.FloatTensor and torch.LongTensor
        features = torch.FloatTensor(features)
        target = torch.tensor(target, dtype=torch.long)

        return features, target