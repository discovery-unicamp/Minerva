# This code was adapted from the DiffusionTS implementation in
# https://github.com/imics-lab/biodiffusion

import math
import torch
from torch import nn
from tqdm import tqdm
from typing import Optional
import lightning as L
from minerva.models.nets.time_series.biodiffusion import Unet1D_cls_free


class BioDiffusion(L.LightningModule):
    """
    Diffusion Model for Biological and HAR signals.

    This LightningModule implements the forward and reverse processes of a
    diffusion model tailored for 1D biological signals. It handles noise
    scheduling (linear or cosine), signal padding, and the optimization process.

    Parameters
    ----------
    model : nn.Module
        The core neural network (e.g., UNet) that predicts the noise.
    synth_data_path : str
        Directory to save/load synthetic data.
    noise_steps : int, optional
        Total number of diffusion timesteps (T). Default is 1000.
    schedule_type : str, optional
        Type of noise schedule to use ('linear' or 'cosine'). Default is 'linear'.
    beta_start : float, optional
        Starting value for the beta schedule. Default is 1e-4.
    beta_end : float, optional
        Ending value for the beta schedule. Default is 0.02.
    n_timesteps : int, optional
        The base length of the input signal. Default is 32.
    lr : float, optional
        Learning rate for the optimizer. Default is 3e-4.
    conditional : bool, optional
        Whether the model is conditioned on class labels (Classifier-Free Guidance). Default is True.
    channels : int, optional
        Number of channels in the input signal. Default is 3.
    signal_padding : int, optional
        Amount of padding to add to the signal to ensure dimensionality
        matches network requirements (e.g., powers of 2). Default is 0.
    """

    def __init__(
        self,
        model: nn.Module,
        noise_steps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        n_timesteps: int = 32,
        lr: float = 3e-4,
        conditional: bool = True,
        channels: int = 3,
        signal_padding: int = 0,
    ) -> None:
        super(BioDiffusion, self).__init__()
        self.noise_steps = noise_steps
        self.scheduler_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.n_timesteps = n_timesteps
        self.lr = lr
        self.conditional = conditional
        self.signal_padding = signal_padding

        # Padding logic to handle specific dimensional requirements
        self.add_signal_pad = signal_padding > 0
        self.left_pad = signal_padding // 2
        self.right_pad = math.ceil(signal_padding / 2)

        # Precompute diffusion process hyperparameters (Alphas and Betas)
        self.betas = self.prepare_noise_schedule()
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.alphas_hat_prev = torch.nn.functional.pad(
            self.alphas_hat[:-1], (1, 0), value=1.0
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_hat_prev) / (1.0 - self.alphas_hat)
        )

        self.is_conditional = conditional
        self.channels = channels
        self.model = model

    def prepare_noise_schedule(self) -> torch.Tensor:
        """
        Calculates the beta variance schedule for the diffusion process.

        Returns
        -------
        torch.Tensor
            A 1D tensor of shape (noise_steps,) containing the beta values.
        """
        if self.scheduler_type == "linear":
            return torch.linspace(
                self.beta_start, self.beta_end, self.noise_steps
            ).float()
        if self.scheduler_type == "cosine":
            s = 0.008  # Arbitrary parameter as selected by https://openreview.net/pdf?id=-NEXDKk8gZ
            steps = self.noise_steps + 1
            x = torch.linspace(0, self.noise_steps, steps, dtype=torch.float64)
            alphas_cumprod = (
                torch.cos(((x / self.noise_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            # Quote from article 'we clip betas to be no larger than 0.999 to prevent singularities
            # at the end of the diffusion process near t = T'
            return torch.clip(betas, 0, 0.999).float()
        else:
            raise ValueError(f"Noise scheduler type {self.scheduler_type} unknown")

    def forward(self, inputs, t, labels=None, with_cond_drop: bool = True):
        # 50% chance to drop condition for Classifier-Free Guidance during training
        cond_drop_prob = 0.5 if with_cond_drop else 0

        if labels is not None:
            return self.model(inputs, t, labels, cond_drop_prob=cond_drop_prob)
        else:
            return self.model(inputs, t, cond_drop_prob=cond_drop_prob)

    def pad_signal(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(
            signal, (self.left_pad, self.right_pad), mode="constant", value=0
        )

    def unpad_signal(self, signal: torch.Tensor) -> torch.Tensor:
        shape = signal.shape
        signal = signal[:, :, self.left_pad : shape[-1] - self.right_pad]
        return signal

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch):
        inputs, labels = batch
        bs = inputs.shape[0]
        device = inputs.device

        if self.add_signal_pad:
            inputs = self.pad_signal(inputs)

        # Sample a random timestep for each sequence in the batch
        t = torch.randint(low=1, high=self.noise_steps, size=(bs,))

        # Generate Gaussian noise to add to the inputs
        noise = torch.randn_like(inputs)

        # Forward diffusion process: q(x_t | x_0)
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None, None].to(device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[t])[:, None, None].to(
            device
        )
        noise_input = sqrt_alpha_hat * inputs + sqrt_one_minus_alpha_hat * noise

        # Predict the noise using the neural network
        if self.is_conditional:
            pred_noise = self.forward(noise_input, t.to(device), labels.squeeze(dim=-1))
        else:
            pred_noise = self.forward(noise_input, t.to(device))

        if self.add_signal_pad:
            pred_noise = self.unpad_signal(pred_noise)
            noise = self.unpad_signal(noise)

        # Calculate loss between true noise and predicted noise
        loss = torch.nn.functional.l1_loss(pred_noise, noise)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def sample(
        self,
        batch_size: int,
        activity: int = 0,
        cond_scale: float = 3.0,
        end: int = 0,
    ):
        """
        Generates new samples using the reverse diffusion process.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        activity : int, optional
            The class label to condition on. If < 0, unconditional generation is used.
        cond_scale : float, optional
            Classifier-Free Guidance scale. Determines how strongly to condition
            the generation on the label. Default is 3.0.
        end : int, optional
            The timestep to stop the reverse process. Default is 0 (full generation).

        Returns
        -------
        torch.Tensor
            The generated signal tensor.
        """
        is_conditional = activity >= 0
        if is_conditional:
            labels = torch.ones((batch_size,)).int().to(self.device) * activity

        with torch.inference_mode():
            # Start from pure Gaussian noise
            x_t = torch.randn(
                (
                    batch_size,
                    self.channels,
                    self.n_timesteps + self.left_pad + self.right_pad,
                )
            ).to(self.device)

            p_bar = tqdm(
                reversed(range(end, self.noise_steps)),
                total=self.noise_steps,
                desc="Sampling step ",
            )

            # Reverse process loop
            for i in p_bar:
                t = (torch.ones(batch_size) * i).int().to(self.device)

                # Predict noise, using Classifier-Free Guidance if conditional
                if is_conditional:
                    cond_pred_noise = self.forward(x_t, t.to(self.device), labels)
                    uncond_pred_noise = self.forward(x_t, t.to(self.device))
                    pred_noise = (
                        1 + cond_scale
                    ) * cond_pred_noise - cond_scale * uncond_pred_noise
                else:
                    pred_noise = self.forward(x_t, t)

                # Fetch hyperparams for current timestep
                alpha = self.alphas.to(self.device)[t][:, None, None]
                alpha_hat = self.alphas_hat.to(self.device)[t][:, None, None]
                alpha_hat_prev = self.alphas_hat_prev.to(self.device)[t][:, None, None]
                beta = self.betas.to(self.device)[t][:, None, None]

                # Estimate x_0 from x_t
                x_start = (
                    torch.sqrt(1.0 / alpha_hat) * x_t
                    - torch.sqrt(1.0 / alpha_hat - 1) * pred_noise
                )
                x_start = torch.clamp(x_start, -4.0, 4.0)

                # Compute posterior mean and variance to step back to x_{t-1}
                posterior_mean = (
                    beta * torch.sqrt(alpha_hat_prev) / (1.0 - alpha_hat)
                ) * x_start + (
                    (1.0 - alpha_hat_prev) * torch.sqrt(alpha) / (1.0 - alpha_hat)
                ) * x_t
                posterior_log_variance = torch.log(
                    self.posterior_variance.to(self.device)[t].clamp(min=1e-20)
                )[:, None, None]

                # Add noise if not the final step
                noise = torch.randn_like(x_t)
                mask_last_step = 0 if i == 0 else 1
                x_t = (
                    posterior_mean
                    + (0.5 * posterior_log_variance).exp() * noise * mask_last_step
                )

            if self.add_signal_pad:
                x_t = self.unpad_signal(x_t)

        return x_t

    def full_forward(
        self, inputs, t, with_cond_drop: bool = True, pass_strategy: str = "single"
    ):

        cond_drop_prob = 0.5 if with_cond_drop else 0
        if self.add_signal_pad and pass_strategy == "double":
            inputs = self.pad_signal(inputs)

        x_t = inputs
        time_step = (torch.ones(x_t.shape[0]) * t).int().to(x_t.device)

        pred_noise = self.model.full_forward(
            x_t, time_step, cond_drop_prob=cond_drop_prob
        )

        alpha = self.alphas.to(x_t.device)[time_step][:, None, None]
        alpha_hat = self.alphas_hat.to(x_t.device)[time_step][:, None, None]
        alpha_hat_prev = self.alphas_hat_prev.to(x_t.device)[time_step][:, None, None]
        beta = self.betas.to(x_t.device)[time_step][:, None, None]

        x_start = (
            torch.sqrt(1.0 / alpha_hat) * x_t
            - torch.sqrt(1.0 / alpha_hat - 1) * pred_noise
        )
        x_start = torch.clamp(x_start, -4.0, 4.0)
        posterior_mean = (
            beta * torch.sqrt(alpha_hat_prev) / (1.0 - alpha_hat)
        ) * x_start + (
            (1.0 - alpha_hat_prev) * torch.sqrt(alpha) / (1.0 - alpha_hat)
        ) * x_t
        posterior_log_variance = torch.log(
            self.posterior_variance.to(x_t.device)[time_step].clamp(min=1e-20)
        )[:, None, None]

        noise = torch.randn_like(x_t)
        mask_last_step = 0 if t == 0 else 1
        x_t = (
            posterior_mean
            + (0.5 * posterior_log_variance).exp() * noise * mask_last_step
        )

        if self.add_signal_pad:
            x_t = self.unpad_signal(x_t)

        return x_t

    def simple_forward(
        self,
        inputs,
        t,
        with_cond_drop: bool = True,
        target_block: int = 4,
        pass_strategy: str = "single",
    ):
        cond_drop_prob = 0.5 if with_cond_drop else 0
        if self.add_signal_pad and pass_strategy == "double":
            inputs = self.pad_signal(inputs)

        x_t = inputs
        time_step = (torch.ones(x_t.shape[0]) * t).int().to(x_t.device)

        x = self.model.simple_forward(
            x_t,
            time=time_step,
            classes=None,
            cond_drop_prob=cond_drop_prob,
            target_block=target_block,
        )

        return x

    def get_init_config(self):
        return {
            "model": Unet1D_cls_free(**self.model.get_init_config()),
            "noise_steps": self.noise_steps,
            "schedule_type": self.scheduler_type,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "n_timesteps": self.n_timesteps,
            "lr": self.lr,
            "conditional": self.is_conditional,
            "channels": self.channels,
            "signal_padding": self.signal_padding,
        }
