# This code was adapted from the DiffusionTS implementation in
# https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Models/interpretable_diffusion/gaussian_diffusion.py
import math
import torch
import torch.nn.functional as F
import lightning as L

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from minerva.models.nets.time_series.diffusionts.diffusionts_model_utils import (
    default,
    identity,
    extract,
)
from minerva.schedulers.diffusionts_lr_sch import (
    ReduceLROnPlateauWithWarmup,
)
from typing import Optional
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# gaussian diffusion trainer class


def linear_beta_schedule(timesteps):
    """Return linearly spaced noise variances scaled by the diffusion step count."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionTS(L.LightningModule):
    """
    Adapts the DiffusionTS model, as described in https://openreview.net/pdf?id=4h1apFjO99
    and implemented in https://github.com/Y-debug-sys/Diffusion-TS, aligned to the
    Lightning pipeline. As described in the paper, its inner model should decompose a time
    series into trend and seasonality components. Ideally, it must be the Transformer class
    implemented in minerva.models.nets.time_series.diffusionts.diffusionts_transformer.py.
    """

    def __init__(
        self,
        model: nn.Module,
        seq_length: int = 60,
        feature_size: int = 6,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        loss_type: str = "l1",
        beta_schedule: str = "cosine",
        eta: float = 0.0,
        use_ff: bool = True,
        reg_weight: float = None,
        # From trainer in
        # https://github.com/Y-debug-sys/Diffusion-TS/blob/main/engine/solver.py
        max_training_steps: Optional[int] = None,
        base_lr: float = 1.0e-5,
        gradient_accumulate_every: int = 2,
        ema_decay: float = 0.995,
        ema_update_interval: int = 10,
        scheduler_hparams: dict = {},
    ):
        """
        DiffusionTS model. The model parameter must be a nn.Module that receives a time series
        and a timestep as input, adds noise according to the timestep, and outputs a time series

        Parameters
        ----------
        model : torch.nn.Module
            Inner model. Must decompose a time series into trend and seasonality components.
        seq_length : int
            The length of the input data.
        feature_size : int
            The number of channels of the input data.
        timesteps :  int
            The number of diffusion timesteps for model training, by default 1000.
        sampling_timesteps : int, optional
            The number of diffusion timesteps applied in synthetic data generation, by default
            None. Must be smaller than or equal to the timesteps for training. If None, it
            copies the value of timesteps for training.
        loss_type: str
            The loss function, by default "l1". Can be either "l1" or "l2" for l1_loss or
            mse_loss from torch.nn.functional, respectively.
        beta_schedule: str
            The function that defines the beta values for every diffusion timestep, by default
            "cosine". Can be either "cosine" or "linear".
        eta: float
            Parameter used in the fast diffusion process, when sampling timesteps is smaller
            than or equal the training timesteps, by default 0.0.
        use_ff: bool
            Whether to use or ignore the fourier loss, by default True. If False, only basic
            loss function (l1_loss or mse_loss) is used.
        reg_weight: float, optional
            Weight of the fourier loss, by default None. If None, reg_weight defaults to a
            fifth of the squared root of the sequence length.
        max_training_steps: int, optional
            Maximum number of training steps, by default None. If None, the model stops training
            when the Lightning trainer's max_epochs is reached.
        base_lr: float
            Initial learning rate, by default 0.00001.
        gradient_accumulate_every: int
            Number of steps to wait before executing an optimizer step, which updates the model
            parameters. By default 2. A value of 2 effectively reduces the total training steps
            by half, so greater values for the total number of training steps must be considered.
        ema_decay: float
            Decay of the Exponential Moving Average shadowing for the model, by default 0.9995.
        ema_update_interval: int
            Update interval of the Exponential Moving Average shadowing for the model, by
            default 10.
        scheduler_params: dict
            Parameters for the scheduler. Must contain the values for the custom scheduler
            ReduceLROnPlateauWithWarmup: "mode", "factor", "patience", "threshold",
            "threshold_mode", "cooldown", "min_lr", "eps", "verbose", "warmup_lr", and "warmup".
        """
        super(DiffusionTS, self).__init__()
        self.model = model
        self.automatic_optimization = False
        self.gradient_accumulate_counter = 0
        self.total_loss = 0
        self.step_counter = 0
        self.max_steps = max_training_steps
        self.base_lr = base_lr
        self.gradient_accumulate_every = gradient_accumulate_every
        self.ema_decay = ema_decay
        self.ema_update_interval = ema_update_interval
        self.scheduler_hparams = scheduler_hparams
        self.last_logged_total_loss = 0
        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate reweighting

        register_buffer(
            "loss_weight",
            torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100,
        )

    def training_step(self, batch, batch_idx):
        """
        Training step. Accumulates gradients according to gradient_accumulate_every, executes
        an optimizer step to update the model parameters, a scheduler step to update the learning
        rate, and updates the Exponential Moving Average shadowing of the model.
        """
        # If the batch is a tuple or list, get the first element
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        loss = self.forward(batch, target=batch)
        loss = loss / self.gradient_accumulate_every
        loss.backward()
        self.total_loss += loss.item()
        self.gradient_accumulate_counter += 1
        if self.gradient_accumulate_counter == self.gradient_accumulate_every:
            self.gradient_accumulate_counter = 0
            clip_grad_norm_(self.parameters(), 1.0)
            self.opt.step()
            self.sch.step(self.total_loss)
            self.opt.zero_grad()
            self.ema.update()
            self.log("train_loss", self.total_loss, on_epoch=False, on_step=True)
            self.last_logged_total_loss = self.total_loss
            self.total_loss = 0
            self.step_counter += 1
        return None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Stop training when the manual optimizer-step counter reaches its limit."""
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            self.trainer.should_stop = True

    def configure_optimizers(self):
        """
        Configures the optimizer, the Exponential Moving Average shadowing of the model, and
        the learning rate scheduler.
        """
        self.opt = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.base_lr,
            betas=[0.9, 0.96],
        )
        self.ema = EMA(
            self.model,
            beta=self.ema_decay,
            update_every=self.ema_update_interval,
        ).to(self.device)
        self.sch = ReduceLROnPlateauWithWarmup(self.opt, **self.scheduler_hparams)

    def predict_noise_from_start(self, x_t, t, x0):
        """Recover the implied noise from noisy and predicted clean series."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        """Recover the clean series estimate from noisy input and predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Return the posterior mean, variance, and clipped log variance."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, t, padding_masks=None):
        """Combine the transformer trend and seasonal outputs into a clean estimate."""
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        """Return predicted noise and clean series, optionally clipping the latter."""
        if padding_masks is None:
            padding_masks = torch.ones(
                x.shape[0], self.seq_length, dtype=bool, device=x.device
            )

        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        x_start = self.output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        """Compute reverse-step statistics and the predicted clean series."""
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None):
        """Sample one reverse step and return it with the clean series estimate."""
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn,
                model_mean,
                model_log_variance,
                x,
                t=batched_times,
                model_kwargs=model_kwargs,
            )
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def sample(self, shape):
        """Generate series from Gaussian noise using every reverse diffusion step."""
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            # Previous, clipping removed
            # img, _ = self.p_sample(img, t)
            img, _ = self.p_sample(img, t, clip_denoised=False)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        """Generate series using the configured reduced set of diffusion timesteps."""
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        """Generate a batch with the configured full or accelerated sampler.

        Parameters
        ----------
        batch_size : int, optional
            Number of series to generate, by default 16.
        model_kwargs : dict, optional
            Keyword arguments passed to the conditioning function.
        cond_fn : callable, optional
            Function returning a guidance gradient from ``x``, ``t``, and keyword arguments.

        Returns
        -------
        torch.Tensor
            Generated series with shape ``(batch_size, seq_length, feature_size)``."""
        feature_size, seq_length = self.feature_size, self.seq_length
        if cond_fn is not None:
            model_kwargs = {} if model_kwargs is None else model_kwargs
            sample_fn = (
                self.fast_sample_cond if self.fast_sampling else self.sample_cond
            )
            return sample_fn(
                (batch_size, seq_length, feature_size),
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))

    @property
    def loss_fn(self):
        """Select the elementwise L1 or L2 reconstruction loss."""
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def q_sample(self, x_start, t, noise=None):
        """Add timestep-dependent Gaussian noise to a clean series."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        """Compute weighted reconstruction loss with optional Fourier-domain loss."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self.output(x, t, padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction="none")

        fourier_loss = torch.tensor([0.0])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(
                torch.real(fft1), torch.real(fft2), reduction="none"
            ) + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction="none")
            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, "b ... -> b (...)", "mean")
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, **kwargs):
        """Validate series dimensions and compute loss at random diffusion timesteps."""
        (
            b,
            c,
            n,
            device,
            feature_size,
        ) = (
            *x.shape,
            x.device,
            self.feature_size,
        )
        assert n == feature_size, f"number of variable must be {feature_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        """Return trend, seasonality, residual, and noisy input at the chosen timestep."""
        (
            b,
            c,
            n,
            device,
            feature_size,
        ) = (
            *x.shape,
            x.device,
            self.feature_size,
        )
        assert n == feature_size, f"number of variable must be {feature_size}"
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(
        self,
        shape,
        target,
        sampling_timesteps,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """Fill missing values with accelerated sampling and Langevin refinement.

        Parameters
        ----------
        shape : tuple of int
            Output shape ``(batch, time, features)`` matching ``target``.
        target : torch.Tensor
            Reference series containing the observed values.
        sampling_timesteps : int
            Number of reverse sampling steps.
        partial_mask : torch.Tensor
            Boolean mask with True entries marking observed values to preserve.
        clip_denoised : bool, optional
            Clip clean predictions to [-1, 1], by default True.
        model_kwargs : dict
            Arguments for ``langevin_fn``, including ``coef`` and ``learning_rate``.

        Returns
        -------
        torch.Tensor
            Completed series retaining the observed target values."""
        if partial_mask is None:
            raise ValueError("partial_mask is required for infill")
        model_kwargs = {} if model_kwargs is None else model_kwargs
        batch, device, total_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(
            time_pairs, desc="conditional sampling loop time step"
        ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(
                sample=img,
                mean=pred_mean,
                sigma=sigma,
                t=time_cond,
                tgt_embs=target,
                partial_mask=partial_mask,
                **model_kwargs,
            )
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape,
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        if partial_mask is None:
            raise ValueError("partial_mask is required for infill")
        model_kwargs = {} if model_kwargs is None else model_kwargs
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="conditional sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample_infill(
                x=img,
                t=t,
                clip_denoised=clip_denoised,
                target=target,
                partial_mask=partial_mask,
                model_kwargs=model_kwargs,
            )

        img[partial_mask] = target[partial_mask]
        return img

    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """Take a reverse step, refine missing values, and restore noisy observations."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(
            sample=pred_img,
            mean=model_mean,
            sigma=sigma,
            t=batched_times,
            tgt_embs=target,
            partial_mask=partial_mask,
            **model_kwargs,
        )

        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.0,
    ):
        """Refine missing entries by optimizing reconstruction of observed values."""
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = (
                        coef * ((mean - input_embs_param) ** 2 / 1.0).mean(dim=0).sum()
                    )
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = (
                        coef
                        * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
                    )
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss / sigma.mean()).mean(dim=0).sum()

                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter(
                    (
                        input_embs_param.data + coef_ * sigma.mean().item() * epsilon
                    ).detach()
                )

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample

    def condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = mean.float() + torch.exp(log_variance) * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self.predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        pred_xstart = self.predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart

    def sample_cond(self, shape, clip_denoised=True, model_kwargs=None, cond_fn=None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, x_start = self.p_sample(
                img,
                t,
                clip_denoised=clip_denoised,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
        return img

    def fast_sample_cond(
        self, shape, clip_denoised=True, model_kwargs=None, cond_fn=None
    ):
        """Generate series with accelerated sampling and a conditioning gradient."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if cond_fn is not None:
                _, x_start = self.condition_score(
                    cond_fn, x_start, img, time_cond, model_kwargs=model_kwargs
                )
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img


if __name__ == "__main__":
    pass
