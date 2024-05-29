import torch


class GRUEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 100,
        in_channels: int = 6,
        encoding_size: int = 10,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        """Gate Recurrent Unit (GRU) Encoder.
        This class is a wrapper for the GRU layer (torch.nn.GRU) followed by a
        linear layer, in order to obtain a fixed-size encoding of the input
        sequence.

        The input sequence is expected to be of shape
        [batch_size, in_channel, seq_len].
        For instance, for HAR data in MotionSense Dataset:
            - in_channel = 6 (3 for accelerometer and 3 for gyroscope); and
            - seq_len = 60 (the number of time steps).

        In forward pass, the input sequence is permuted to
        [seq_len, batch_size, in_channel] before being fed to the GRU layer.
        The output of forward pass is the encoding of shape
        [batch_size, encoding_size].

        Parameters
        ----------
        hidden_size : int, optional
            The number of features in the hidden state of the GRU,
            by default 100
        in_channel: int, optional
            The number of input features (e.g. 6 for HAR data in MotionSense
            Dataset), by default 6
        encoding_size : int, optional
            Size of the encoding (output of the linear layer).
        num_layers : int, optional
            Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. By default 1
        dropout : float, optional
            If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional : bool, optional
            If ``True``, becomes a bidirectional GRU, by default True
        """
        super().__init__()
        
        # Parameters 
        self.hidden_size = hidden_size
        self.in_channel = in_channels
        self.num_layers = num_layers
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        # If bidirectional is true, the number of directions is 2
        self.num_directions = 2 if bidirectional else 1

        # Instantiate the GRU layer
        self.rnn = torch.nn.GRU(
            input_size=self.in_channel,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Instantiate the linear leayer
        # If bidirectional is true, the input of linear layer is
        # hidden_size * 2 (because the output of GRU is concatenated)
        # Otherwise, the input of linear layer is hidden_size
        self.nn = torch.nn.Linear(
            self.hidden_size * self.num_directions, self.encoding_size
        )

    def forward(self, x):
        # Permute the input sequence from [batch_size, in_channel, seq_len]
        # to [seq_len, batch_size, in_channel]
        x = x.permute(2, 0, 1)

        # The initial hidden state (h0) is set to zeros of shape
        # [num_layers * num_directions, batch_size, hidden_size]
        initial_state = torch.zeros(
            self.num_layers * self.num_directions,  # initial_state.shape[0]
            x.shape[1],                             # initial_state.shape[1]
            self.hidden_size,                       # initial_state.shape[2]
            device=x.device,
            # requires_grad=False          # This is not a learnable parameter
        )

        # Forward pass of the GRU layer
         # out shape = [seq_len, batch_size, num_directions*hidden_size]
        out, _ = self.rnn(
            x, initial_state
        ) 
        
        # Pick the last state returned by the GRU of shape
        # [batch_size, num_directions*hidden_size] and squeeze it (remove the
        # first dimension if the size is 1)
        out = out[-1].squeeze(0)
        
        # Pass the output of GRU to the linear layer to obtain the encodings
        encodings = self.nn(out)
        
        # encodings shape = [batch_size, encoding_size]
        return encodings
