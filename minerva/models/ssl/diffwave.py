# This code was adapted from the DiffWave implementation in
# https://github.com/philsyn/DiffWave-unconditional

import torch
import torch.nn as nn
import lightning as L
from typing import Optional
from minerva.models.nets.time_series.diffwave import Conv, ZeroConv1d, ResidualGroup


class DiffWave(L.LightningModule):
    """DiffWave model adapted for time-series sensor data (e.g. IMU).

    Implements a diffusion-based generative model capable of synthesizing
    multi-channel temporal signals conditioned on class labels.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (e.g., sensor axes). Default is 1.
    res_channels : int, optional
        Number of residual channels in convolutional blocks. Default is 256.
    skip_channels : int, optional
        Number of skip channels. Default is 128.
    out_channels : int, optional
        Number of output channels. Default is 1.
    num_res_layers : int, optional
        Number of residual layers. Default is 30.
    dilation_cycle : int, optional
        Dilation reset cycle. Default is 10.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 2e-4.
    T : int, optional
        Number of diffusion timesteps. Default is 200.
    beta_0 : float, optional
        Starting β value for the noise schedule. Default is 0.0001.
    beta_T : float, optional
        Final β value for the noise schedule. Default is 0.02.
    conditional : bool, optional
        Whether to use label conditioning. Default is False.
    num_classes : int, optional
        Number of conditioning classes. Default is 6.
    """

    def __init__(
        self,
        in_channels: int = 1,
        res_channels: int = 256,
        skip_channels: int = 128,
        out_channels: int = 1,
        num_res_layers: int = 30,
        dilation_cycle: int = 10,
        diffusion_step_embed_dim_in: int = 128,
        diffusion_step_embed_dim_mid: int = 512,
        diffusion_step_embed_dim_out: int = 512,
        learning_rate: float = 2e-4,
        T: int = 200,
        beta_0: float = 0.0001,
        beta_T: float = 0.02,
        conditional: bool = False,
        num_classes: int = 6,
    ):
        """Build the residual denoiser, diffusion schedule, and optional label embedding."""
        super(DiffWave, self).__init__()

        self.learning_rate = learning_rate
        self.conditional = conditional
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.num_res_layers = num_res_layers
        self.dilation_cycle = dilation_cycle
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.diffusion_step_embed_dim_mid = diffusion_step_embed_dim_mid
        self.diffusion_step_embed_dim_out = diffusion_step_embed_dim_out
        self.out_channels = out_channels
        self.T = T
        self.beta_0 = beta_0
        self.beta_T = beta_T
        self.num_classes = num_classes

        # initial conv1x1 with relu
        self.init_conv = nn.Sequential(
            Conv(in_channels, res_channels, kernel_size=1), nn.ReLU()
        )

        # all residual layers
        self.residual_layer = ResidualGroup(
            res_channels=res_channels,
            skip_channels=skip_channels,
            num_res_layers=num_res_layers,
            dilation_cycle=dilation_cycle,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
        )

        # final conv1x1 -> relu -> zeroconv1x1
        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, out_channels),
        )
        self.global_emb = nn.Embedding(num_classes, 128)

        self.diffusion_hyperparams = calc_diffusion_hyperparams(T, beta_0, beta_T)

    def get_init_config(self):
        """Return constructor settings used to rebuild the DiffWave model."""
        return {
            "in_channels": self.in_channels,
            "res_channels": self.res_channels,
            "skip_channels": self.skip_channels,
            "out_channels": self.out_channels,
            "num_res_layers": self.num_res_layers,
            "dilation_cycle": self.dilation_cycle,
            "diffusion_step_embed_dim_in": self.diffusion_step_embed_dim_in,
            "diffusion_step_embed_dim_mid": self.diffusion_step_embed_dim_mid,
            "diffusion_step_embed_dim_out": self.diffusion_step_embed_dim_out,
            "T": self.T,
            "beta_0": self.beta_0,
            "beta_T": self.beta_T,
            "conditional": self.conditional,
            "num_classes": self.num_classes,
        }

    def forward(self, input_data, label: Optional[int | torch.Tensor] = None):
        """Predict noise from a signal and timestep, optionally conditioned on labels."""
        input, diffusion_steps = input_data
        label_emb = None
        if self.conditional and label is not None:
            label_emb = self.global_emb(label)  # shape: (B, 128)
        x = input
        x = self.init_conv(x)
        x = self.residual_layer((x, diffusion_steps), label_emb)
        x = self.final_conv(x)

        return x

    def training_step(self, batch, batch_idx):
        """Compute and log noise-prediction loss for a signal-label batch."""
        X, Y = batch
        X = X.to(self.device, non_blocking=True)
        Y = Y.to(self.device, non_blocking=True)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss = self.training_loss(nn.MSELoss(), X, label=Y)  # compute training loss
        self.log("train_loss", loss)
        # loss.backward()
        # optimizer.step()
        return loss

    def configure_optimizers(self):
        """Return an Adam optimizer with the configured learning rate."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]

    def training_loss(self, loss_fn, X, label: Optional[int | torch.Tensor] = None):
        """
        Compute the training loss of epsilon and epsilon_theta

        Parameters:
        net (torch network):            the wavenet model
        loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
        X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors

        Returns:
        training loss
        """
        device = X.device
        _dh = self.diffusion_hyperparams
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

        input = X
        B, C, L = input.shape  # B is batchsize, C=1, L is input length

        diffusion_steps = torch.randint(
            T, size=(B, 1, 1), device=device
        )  # randomly sample diffusion steps from 1~T
        z = self.std_normal(input.shape, device=device)
        Alpha_bar = Alpha_bar.to(device)
        z = z.to(device)
        transformed_X = (
            torch.sqrt(Alpha_bar[diffusion_steps]) * input
            + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
        )  # compute x_t from q(x_t|x_0)
        epsilon_theta = self.forward(
            (transformed_X, diffusion_steps.view(B, 1)), label
        )  # predict \epsilon according to \epsilon_\theta
        return loss_fn(epsilon_theta, z)

    def std_normal(self, size, device=None):
        """
        Generate the standard Gaussian variable of a certain size
        """

        return torch.normal(0, 1, size=size, device=device)

    def sample(
        self,
        batch_size: int = 1,
        segment_length: int = 16000,
        condition: Optional[int | torch.Tensor] = None,
    ):
        """
        Generate synthetic samples using the DiffWave model.

        This function performs the reverse denoising process to generate
        data from initial noise, optionally conditioned on a class label.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        segment_length : int
            Temporal length of each sample (number of timesteps).
        condition : int or torch.Tensor, optional
            Class label for conditional generation. Can be:
                - int: same class for all samples
                - torch.Tensor: tensor of labels
                - None: unconditional generation

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, in_channels, segment_length),
            containing the generated synthetic samples.
        """
        output_shape = (batch_size, self.in_channels, segment_length)

        _dh = self.diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = (
            _dh["T"],
            _dh["Alpha"],
            _dh["Alpha_bar"],
            _dh["Sigma"],
        )
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert len(output_shape) == 3

        if condition is not None:
            if isinstance(condition, int):
                condition = torch.tensor([condition] * batch_size, dtype=torch.long)
            elif isinstance(condition, torch.Tensor) and condition.ndim == 0:
                condition = condition.unsqueeze(0)
            condition = condition.to(self.device)
        x = self.std_normal(output_shape, device=self.device)
        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                diffusion_steps = (t * torch.ones((output_shape[0], 1))).to(
                    self.device
                )  # use the corresponding reverse step
                epsilon_theta = self.forward(
                    (x, diffusion_steps), condition
                )  # predict \epsilon according to \epsilon_\theta
                x = (
                    x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
                ) / torch.sqrt(
                    Alpha[t]
                )  # update x_{t-1} to \mu_\theta(x_t)
                if t > 0:
                    x = x + Sigma[t] * self.std_normal(
                        output_shape, device=self.device
                    )  # add the variance term to x_{t-1}
        return x

    def simple_forward(
        self,
        input: torch.Tensor,
        label: Optional[int | torch.Tensor] = None,
        target_time_step: int = 0,
        target_res_layer: Optional[int] = None,
        return_skip: bool = False,
        flatten: bool = False,
    ):
        """
        Forward pass for feature extraction at a specific diffusion timestep
        and residual layer.

        Allows extracting:
        - the residual activation `h` at a chosen residual block, or the accumulated skip connection vector.

        This is the recommended method for obtaining embeddings for downstream
        tasks such as classification, clustering, or visualization (e.g., t-SNE).

        Parameters
        ----------
        input : torch.Tensor
            Input signal tensor of shape (B, C_in, L).
        label : int or torch.Tensor, optional
            Class label(s) for conditional embedding.
            - int: applied to all samples in the batch
            - tensor of shape (B,)

        target_time_step : int, optional
            Diffusion timestep to embed. Default is 0 (early denoising step).

        target_res_layer : int or None, optional
            If provided, returns the intermediate residual activation `h`
            **at the specific residual block index**.
            If None, the final block’s output is returned.

        return_skip : bool, optional
            If False (default), returns the residual activation `h`.
            If True, returns the accumulated skip-connection vector.

        Returns
        -------
        torch.Tensor
            A feature vector representing the input, with shape:
                - (B, res_channels) if returning `h`
                - (B, skip_channels) if returning skip output
            The temporal dimension is removed by global averaging.

        Notes
        -----
        - Output is averaged across time using `mean(dim=-1)` to produce
          a fixed-length embedding.
        """

        B, C, L = input.shape
        diffusion_steps = (target_time_step * torch.ones((B, 1))).to(input.device)
        label_emb = None
        if self.conditional and label is not None:
            label_emb = self.global_emb(label)  # shape: (B, 128)
        x = input
        x = self.init_conv(x)
        out_res, skip = self.residual_layer.forward_emb(
            (x, diffusion_steps),
            label_emb=label_emb,
            target_res_layer=target_res_layer,
        )
        output = out_res if not return_skip else skip
        if output.dim() == 3:
            if not flatten:
                output = output.mean(dim=-1)  # (B, C)
            else:
                output = torch.flatten(output, start_dim=1)
            # output.reshape(output.size(0), -1)
        return output

    def full_forward(
        self,
        input: torch.Tensor,
        label: Optional[int | torch.Tensor] = None,
        target_time_step: int = 0,
    ):
        """
        Full forward pass that reproduces the complete denoising operation of
        DiffWave at a specific diffusion timestep.

        It returns the fully denoised output for the given timestep, not an
        intermediate latent embedding.

        Parameters
        ----------
        input : torch.Tensor
            Noisy input sample at diffusion timestep t.
            Shape: (B, C_in, L).
        label : int or torch.Tensor, optional
            Optional class conditioning label(s). Used only if the model
            was initialized with ``conditional=True``.
        target_time_step : int, optional
            Diffusion timestep t from which denoising will occur.
            Default is 0 (final denoising step).

        Returns
        -------
        torch.Tensor
            Denoised output after applying the final DiffWave step.
            Shape: (B, out_channels, L).
        """

        _dh = self.diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = (
            _dh["T"],
            _dh["Alpha"],
            _dh["Alpha_bar"],
            _dh["Sigma"],
        )
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T

        B, C, L = input.shape
        diffusion_steps = (target_time_step * torch.ones((B, 1))).to(input.device)
        output_shape = (B, C, L)

        label_emb = None
        if self.conditional and label is not None:
            label_emb = self.global_emb(label)  # shape: (B, 128)
        x = input
        x_t = input
        x = self.init_conv(x)
        x = self.residual_layer((x, diffusion_steps), label_emb=label_emb)
        output = self.final_conv(x)

        x_t = (
            x_t
            - (1 - Alpha[target_time_step])
            / torch.sqrt(1 - Alpha_bar[target_time_step])
            * output
        ) / torch.sqrt(
            Alpha[target_time_step]
        )  # update x_{t-1} to \mu_\theta(x_t)

        if target_time_step > 0:
            x_t = x_t + Sigma[target_time_step] * self.std_normal(
                output_shape, device=input.device
            )  # add the variance term to x_{t-1}
        return x_t


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] = (
            Alpha_bar[t] * Alpha_bar[t - 1]
        )  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] = (
            Beta_tilde[t] * (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
        )  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
        T,
        Beta,
        Alpha,
        Alpha_bar,
        Sigma,
    )
    diffusion_hyperparams = _dh
    return diffusion_hyperparams
