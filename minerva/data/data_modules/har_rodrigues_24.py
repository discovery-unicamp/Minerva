import numpy as np
import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from lightning import LightningDataModule
from torch.utils.data import DataLoader
#from .data_har_unzip import fetch_and_unzip_dataset

# from http://www.johnvinyard.com/blog/?p=268

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    # dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


def opp_sliding_window(data_x, data_y, ws, ss):
    
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)

# Defining the data loader for the implementation
class HARDatasetCPC(Dataset):
    def __init__(self, root_dir, data_file, input_size, window, overlap, phase):

        self.filename = os.path.join(root_dir, data_file)
        
        self.data_raw = self.load_dataset()
        assert input_size == self.data_raw[phase]['data'].shape[1]
       # print("Data Raw", self.data_raw[phase]['data'].shape)
        # Obtaining the segmented data
        self.data, self.labels = \
            opp_sliding_window(self.data_raw[phase]['data'],
                               self.data_raw[phase]['labels'],
                               window, overlap)
        #print("Data", self.data.shape)
        #print("Labels", self.labels.shape)

    # Load .csv file
    
    def load_dataset (self):

        data_path = Path(self.filename)

        #print(data_path)

        datasets = {}

        for phase in ['train', 'val', 'test']:

            phase_path = data_path / phase

            datas_x = []

            data_y = []

            for f in phase_path.glob('*.csv'):

                data = pd.read_csv(f)

                x = data[['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y', 'gyro-z']].values
                
                datas_x.append(x)

                y = data['activity code'].values

                data_y.append(y)

            datasets[phase] = {'data': np.concatenate(datas_x), 'labels': np.concatenate(data_y)}
            datasets[phase]['data'] = datasets[phase]['data'].astype(np.float32)
            datasets[phase]['labels'] = datasets[phase]['labels'].astype(np.uint8)

        return datasets


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]

        # Aplicar permute para ter a forma [64, 6, 60]
        data = torch.from_numpy(data).float().permute(1, 0)

        label = torch.from_numpy(np.asarray(self.labels[index])).long()
        return data, label

class HARDataModuleCPC(LightningDataModule):
    def __init__(self, root_dir, data_file = "RealWorld_raw", input_size = 6, window = 60, overlap = 30, batch_size = 64):
        super().__init__()
        self.batch_size = batch_size

        #fetch_and_unzip_dataset(root_dir, data_file)
        self.train_dataset = HARDatasetCPC(root_dir, data_file, input_size, window, overlap, phase='train')
        self.val_dataset = HARDatasetCPC(root_dir, data_file, input_size, window, overlap, phase='val')
        self.test_dataset = HARDatasetCPC(root_dir, data_file, input_size, window, overlap, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)