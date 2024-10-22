import numpy as np
import pytest
from minerva.transforms.byol import BYOLTransform

def test_byol_transforms():
    
    # Random 3 channel input
    x = np.random.rand(3, 255, 701)
    
    input_size = 224
    
    transform = BYOLTransform(input_size=input_size,
                            degrees=5,
                            r_prob=0.5,
                            h_prob=0.5,
                            v_prob=0.5,
                            collor_jitter_prob=0.5,
                            grayscale_prob=0.2,
                            gaussian_blur_prob=0.5,
                            solarize_prob=0.0
                            )
    
    y = np.random.rand(3, input_size, input_size)
    
    x_transformed = transform(x)
    
    assert x_transformed[0].shape == y.shape and x_transformed[1].shape == y.shape