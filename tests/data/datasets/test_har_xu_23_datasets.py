from minerva.data.datasets.har_xu_23 import TNCDataset
import numpy as np
import torch

def test_pretext_tnc_dataset():
    # Example configuration
    time_series_data = np.random.randn(100, 6, 1000)  # (n_samples, n_channels, n_timesteps)

    # Instantiate the dataset
    tnc_dataset = TNCDataset(
        x=time_series_data,
    )

    # Retrieve a sample from the dataset
    sample_index = 0
    central_window, close_neighbors, non_neighbors = tnc_dataset[sample_index]
    expected_output_shape_cw = torch.Size([128, 6]) #(window_size,n_channels)
    expected_output_shape_cn = torch.Size([5, 128, 6]) #(mc_sample_size,window_size, n_channels, )
    expected_output_shape_nn = torch.Size([5, 128, 6]) #(mc_sample_size,window_size, n_channels, )

    assert (
        central_window.shape == expected_output_shape_cw
    ), f"Expected output shape {expected_output_shape_cw}, but got {central_window.shape}"
    
    assert (
        close_neighbors.shape == expected_output_shape_cn
    ), f"Expected output shape {expected_output_shape_cn}, but got {close_neighbors.shape}"
    
    assert (
        non_neighbors.shape == expected_output_shape_nn
    ), f"Expected output shape {expected_output_shape_nn}, but got {non_neighbors.shape}"
    