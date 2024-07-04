import torch
import torch.nn as nn
from typing import Tuple

class TFC_Conv_Backbone(nn.Module):
    def _calculate_fc_input_features(self, backbone: torch.nn.Module, input_shape: Tuple[int, int]) -> int:
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)
    
    def __init__(self, input_channels, TS_length, single_encoding_size = 128):
        super(TFC_Conv_Backbone, self).__init__()
        self.conv_block_t = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 60, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block_f = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 60, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_t = nn.Sequential(
            nn.Linear(self._calculate_fc_input_features(self.conv_block_t, (input_channels, TS_length)), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, single_encoding_size)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(self._calculate_fc_input_features(self.conv_block_t, (input_channels, TS_length)), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, single_encoding_size)
        )

    def forward(self, x_in_t, x_in_f):
        x = self.conv_block_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)
        z_time = self.projector_t(h_time)

        f = self.conv_block_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
    

class TFC_PredicionHead(nn.Module):
    def __init__(self, num_classes, connections=2, single_encoding_size=128):
        super(TFC_PredicionHead, self).__init__()
        if connections != 2:
            print(f"Only one pipeline is on: {connections} connections.")
        self.logits = nn.Linear(connections*single_encoding_size, 64)
        self.logits_simple = nn.Linear(64, num_classes)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
