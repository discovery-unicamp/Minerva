# Implementation of the Latent Diffusion Model (LDM) for time series data, 
# specifically designed for multi-channel sensor signals (e.g., IMU data). 
# https://github.com/compvis/latent-diffusion

import torch
import torch.nn as nn
import numpy as np
import lightning as L
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from lightning.pytorch.utilities import rank_zero_only
from minerva.models.nets.time_series.ts_ldm import UNetModel1d
from minerva.models.loaders import FromPretrained

from typing import Union

def exists(val):
    """Checks whether a given value is not None.

    Parameters
    ----------
    val : Any
        Input object or tensor.

    Returns
    -------
    bool
        True if `val` is not None, False otherwise.
    """
    return val is not None

def default(val, d):
    """Returns the provided value if it exists, otherwise computes or returns a default.

    Parameters
    ----------
    val : Any
        Primary value to check.
    d : Any or Callable
        Default value or factory function to execute if `val` is None.

    Returns
    -------
    Any
        Evaluated result or default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def extract_into_tensor(a, t, x_shape):
    """Extracts elements from a 1D tensor `a` at indices `t` and reshapes the output
    to broadcast across `x_shape`.

    Parameters
    ----------
    a : torch.Tensor
        1D source lookup tensor.
    t : torch.Tensor
        Index tensor of shape (B,).
    x_shape : torch.Size or tuple
        Target shape for broadcasting alignment.

    Returns
    -------
    torch.Tensor
        Reshaped tensor of shape (B, 1, ..., 1) matching the dimensionality of `x_shape`.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    """Generates standard normal noise matching a target shape, with optional channel repeating.

    Parameters
    ----------
    shape : tuple or torch.Size
        Target output shape.
    device : torch.device
        Target hardware device.
    repeat : bool, optional
        Whether to generate noise for a single sample and repeat across batch. Default is False.

    Returns
    -------
    torch.Tensor
        Random normal noise tensor matching `shape`.
    """
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """Generates variance schedule $(beta_1, dots, beta_T)$ for diffusion processes.

    Parameters
    ----------
    schedule : str
        Schedule type: ``"linear"``, ``"cosine"``, ``"sqrt_linear"``, or ``"sqrt"``.
    n_timestep : int
        Total number of diffusion timesteps $T$.
    linear_start : float, optional
        Initial beta value for linear schedules. Default is 1e-4.
    linear_end : float, optional
        Final beta value for linear schedules. Default is 2e-2.
    cosine_s : float, optional
        Offset parameter $s$ for cosine schedule. Default is 8e-3.

    Returns
    -------
    np.ndarray
        Array of $\beta_t$ values of shape (n_timestep,).
    """
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

class LitEma(nn.Module):
    """Exponential Moving Average (EMA) manager for model parameters.

    Parameters
    ----------
    model : nn.Module
        Target neural network model whose parameters will be tracked.
    decay : float, optional
        Decay factor for EMA updating. Default is 0.9999.
    use_num_updates : bool, optional
        Whether to adjust decay dynamically based on update step count. Default is True.
    """
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_updates else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        """Updates shadow parameter buffers with current model weights.

        Parameters
        ----------
        model : nn.Module
            Source model with updated parameter values.
        """
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())
            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))

    def copy_to(self, model):
        """Copies shadow parameters into the target model parameters.

        Parameters
        ----------
        model : nn.Module
            Destination model to receive EMA weights.
        """
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)

    def store(self, parameters):
        """Saves current parameter tensors into temporary storage.

        Parameters
        ----------
        parameters : Iterable[nn.Parameter]
            Parameters to clone and store.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """Restores stored parameter tensors back to model.

        Parameters
        ----------
        parameters : Iterable[nn.Parameter]
            Parameters to overwrite with stored values.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

class DDPM(L.LightningModule):
    """Denoising Diffusion Probabilistic Model (DDPM) implementation in PyTorch Lightning.

    Parameters
    ----------
    unet_model : nn.Module
        Epsilon-prediction or score network (e.g. 1D U-Net).
    timesteps : int, optional
        Total diffusion steps $T$. Default is 1000.
    beta_schedule : str, optional
        Variance schedule type. Default is "linear".
    loss_type : str, optional
        Reconstruction loss type (``"l1"`` or ``"l2"``). Default is "l2".
    use_ema : bool, optional
        Whether to track parameters with Exponential Moving Average. Default is True.
    sequence_length : int, optional
        Temporal signal length. Default is 16.
    channels : int, optional
        Signal channel count. Default is 4.
    log_every_t : int, optional
        Logging step interval. Default is 100.
    clip_denoised : bool, optional
        Whether to clip generated samples during reverse steps. Default is True.
    linear_start : float, optional
        Schedule initial $beta$. Default is 1e-4.
    linear_end : float, optional
        Schedule final $beta$. Default is 2e-2.
    cosine_s : float, optional
        Offset parameter for cosine schedule. Default is 8e-3.
    original_elbo_weight : float, optional
        Weight factor for variational lower bound loss. Default is 0.0.
    v_posterior : float, optional
        Posterior variance interpolation factor. Default is 0.0.
    l_simple_weight : float, optional
        Weight factor for simple MSE loss. Default is 1.0.
    parameterization : str, optional
        Model prediction objective (``"eps"`` or ``"x0"``). Default is "eps".
    """
    
    def __init__(self,
                 unet_model: nn.Module,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 use_ema=True,
                 sequence_length=16,
                 channels=4,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 original_elbo_weight=0.,
                 v_posterior=0.,
                 l_simple_weight=1.,
                 parameterization="eps"):
        super().__init__()
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.sequence_length = sequence_length
        self.channels = channels
        self.loss_type = loss_type
        self.beta_schedule = beta_schedule

        # Assign the UNet directly
        self.model = unet_model
        
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(
        self, 
        beta_schedule="linear", 
        timesteps=1000, 
        linear_start=1e-4, 
        linear_end=2e-2, 
        cosine_s=8e-3
    ):
        """Computes and registers diffusion hyperparameters as model buffers.

        Parameters
        ----------
        beta_schedule : str, optional
            Variance schedule type. Default is "linear".
        timesteps : int, optional
            Total diffusion steps. Default is 1000.
        linear_start : float, optional
            Initial $\beta$. Default is 1e-4.
        linear_end : float, optional
            Final $\beta$. Default is 2e-2.
        cosine_s : float, optional
            Cosine schedule offset. Default is 8e-3.
        """
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError()
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

    @contextmanager
    def ema_scope(self):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())

    def q_sample(self, x_start, t, noise=None):
        """Diffuses input data $x_0$ to step $t$ via forward sampling process.

        Parameters
        ----------
        x_start : torch.Tensor
            Clean original input tensor $x_0$.
        t : torch.Tensor
            Diffusion timestep batch tensor.
        noise : torch.Tensor, optional
            Target noise tensor. If None, generated automatically.

        Returns
        -------
        torch.Tensor
            Noisy sample $x_t$ at timestep $t$.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, noise=None):
        """Computes training loss objectives for diffusion step $t$.

        Parameters
        ----------
        x_start : torch.Tensor
            Clean input tensor $x_0$.
        t : torch.Tensor
            Timestep tensor.
        noise : torch.Tensor, optional
            Target Gaussian noise.

        Returns
        -------
        loss : torch.Tensor
            Weighted objective loss scalar.
        loss_dict : dict
            Dictionary containing individual logged loss components.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        target = noise if self.parameterization == "eps" else x_start

        if self.loss_type == 'l1':
            loss = (target - model_out).abs().mean(dim=[1, 2])
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, model_out, reduction='none').mean(dim=[1, 2])
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        # Variational lower bound loss
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        """Forward pass executing noise sampling and loss computation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor batch.

        Returns
        -------
        Tuple[torch.Tensor, dict]
            Computed total loss and logging dictionary.
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.model_ema(self.model)


class TSLatentDiffusion(DDPM):
    """Latent Diffusion Model (LDM) tailored for time series sensor data (e.g., IMU signals).

    Parameters
    ----------
    unet_model : nn.Module
        Instantiated 1D U-Net backbone model.
    first_stage_model : Union[nn.Module, FromPretrained]
        Instantiated first-stage Autoencoder (VAE).
    scale_factor : float, optional
        Latent scaling multiplier. Default is 1.0.
    scale_by_std : bool, optional
        Whether to estimate scale factor dynamically from standard deviation. Default is False.
    unconditional : bool, optional
        Whether model operates unconditionally without class labels. Default is True.
    learning_rate : float, optional
        Optimizer learning rate. Default is 2e-4.
    use_scheduler : bool, optional
        Whether to use linear warmup learning rate schedule. Default is True.
    warmup_steps : int, optional
        Number of warmup optimization steps. Default is 500.
    cycle_length : int, optional
        Decay step length for scheduler. Default is 100000.
    f_start : float, optional
        Initial learning rate multiplier. Default is 1e-6.
    f_max : float, optional
        Maximum learning rate multiplier. Default is 1.0.
    f_min : float, optional
        Final learning rate multiplier. Default is 1.0.
    """
    def __init__(self,
                 unet_model: nn.Module,             # Instantiated UNet model
                 first_stage_model: Union[nn.Module, FromPretrained],      # Instantiated Autoencoder
                 scale_factor=1.0,
                 scale_by_std=False,
                 unconditional=True,
                 learning_rate=2e-4,     # Slightly higher for 1D Unet compared to images
                 use_scheduler=True, 
                 warmup_steps=500,       # Approx 4-5 epochs of warmup for HAR datasets
                 cycle_length=100000,    # Large enough to cover total training steps
                 f_start=1e-6,           # Starting learning rate multiplier
                 f_max=1.0,              # Maximum learning rate multiplier
                 f_min=1.0,              # Final learning rate multiplier (1.0 = no decay)
                 *args, **kwargs):
        super().__init__(unet_model=unet_model, *args, **kwargs)
        
        self.unconditional = unconditional
        self.scale_by_std = scale_by_std
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
            
        # Freeze the VAE (First Stage)
        self.first_stage_model = first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
            
        # Scheduler parameters mapped directly from the original Latent Diffusion YAML
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        self.warmup_steps = warmup_steps
        self.cycle_length = cycle_length
        self.f_start = f_start
        self.f_max = f_max
        self.f_min = f_min

    def get_first_stage_encoding(self, encoder_posterior):
        """Extracts scaled latent sample from first-stage encoder output.

        Parameters
        ----------
        encoder_posterior : Any
            Posterior distribution object or raw tensor representation.

        Returns
        -------
        torch.Tensor
            Scaled latent space tensor.
        """
        if hasattr(encoder_posterior, "sample"):
            z = encoder_posterior.sample()
        else:
            z = encoder_posterior
        return self.scale_factor * z
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        """Dynamically computes latent scaling factor during first batch step to achieve unit variance.

        Parameters
        ----------
        batch : Any
            Training batch tuple or tensor.
        batch_idx : int
            Batch index counter.
        """
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            # Handles batches that are just [x] or [x, y]
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(posterior).detach()
            del self.scale_factor  # Remove any existing scale factor to avoid interference
            # Calculate and register the scale factor
            scale = 1. / z.flatten().std()
            self.register_buffer('scale_factor', scale)
            mode = "UNCONDITIONAL" if self.unconditional else "CONDITIONAL"
            print(f"### LDM 1D ({mode}): Scale factor dynamically adjusted to {self.scale_factor.item():.4f} ###")
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        """Pads physical signal and encodes it into first-stage latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (B, C, L_orig).

        Returns
        -------
        DiagonalGaussianDistribution1d or torch.Tensor
            Encoder posterior distribution or output latent encoding.
        """
        if hasattr(self.first_stage_model, "adapter_pad"):
            x = self.first_stage_model.adapter_pad(x)
            
        return self.first_stage_model.encode(x)

    @torch.no_grad()
    def decode_first_stage(self, z):
        """Decodes latent representation and crops output back to physical signal length.

        Parameters
        ----------
        z : torch.Tensor
            Latent sequence tensor of shape (B, C_lat, L_lat).

        Returns
        -------
        torch.Tensor
            Reconstructed physical signal tensor of shape (B, C, L_orig).
        """
        
        z = z / self.scale_factor
        out = self.first_stage_model.decode(z)
        
        if hasattr(self.first_stage_model, "adapter_unpad"):
            out = self.first_stage_model.adapter_unpad(out)
            
        return out

    def shared_step(self, batch):
        """Executes latent encoding and DDPM loss computation.

        Parameters
        ----------
        batch : Any
            Input batch.

        Returns
        -------
        loss : torch.Tensor
            Total training loss.
        loss_dict : dict
            Dictionary of logged loss values.
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        # Extract the latent space representation
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior)
        
        # Pass the latent (z) to the mother class DDPM forward pass
        loss, loss_dict = self(z)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        """Lightning step for model training.

        Parameters
        ----------
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Training step loss.
        """
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Lightning step for model validation with optional EMA evaluation.

        Parameters
        ----------
        batch : Any
            Validation batch.
        batch_idx : int
            Batch index.
        """
        """Identical to the original block, using ema_scope to validate smoothed weights"""
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        """Configures AdamW optimizer and linear warmup learning rate scheduler.

        Returns
        -------
        list of Optimizer or Tuple[list, list]
            Optimizer and scheduler configurations.
        """
        # 1. Start with the U-Net parameters
        params = list(self.model.parameters())
        
        # 2. Add conditioning model parameters IF we are in Conditional Mode
        #    This allows the model to learn the embeddings jointly with the diffusion process.
        # if not self.unconditional and self.cond_stage_model is not None:
        #     cond_params = list(self.cond_stage_model.parameters())
        #     if len(cond_params) > 0:
        #         print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
        #         params = params + cond_params

        # 3. Initialize AdamW Optimizer
        opt = torch.optim.AdamW(params, lr=self.learning_rate)

        # 4. Learning Rate Scheduler (Linear Warmup & Decay)
        if self.use_scheduler:
            print(f"Setting up LambdaLR scheduler with {self.warmup_steps} warmup steps...")
            
            def lr_lambda(current_step):
                """
                Returns the learning rate multiplier for the current training step.
                """
                # Phase 1: Linear Warmup (scales from f_start to f_max)
                if current_step < self.warmup_steps:
                    f = (self.f_max - self.f_start) / float(max(1, self.warmup_steps)) * current_step + self.f_start
                    return f
                
                # Phase 2: Linear Decay (scales from f_max to f_min over cycle_length)
                # Note: If f_max == f_min (default), this simply maintains the max learning rate.
                else:
                    n = current_step - self.warmup_steps
                    f = self.f_min + (self.f_max - self.f_min) * (self.cycle_length - n) / float(max(1, self.cycle_length))
                    return f
            
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda),
                'interval': 'step', 
                'frequency': 1
            }
            return [opt], [scheduler]
            
        return opt
    
    @torch.no_grad()
    def p_sample(self, z, t, t_index):
        """Performs single-step reverse denoising on latent sequence $z_t$.

        Parameters
        ----------
        z : torch.Tensor
            Latent sequence tensor at timestep $t$.
        t : torch.Tensor
            Timestep tensor.
        t_index : int
            Current loop index $t$.

        Returns
        -------
        torch.Tensor
            Denoised latent sequence $z_{t-1}$.
        """
        # 1. The diffusion model (UNet) predicts the noise (epsilon)
        model_out = self.model(z, t)
        
        # 2. Extract mathematical constants for time step t
        betas_t = extract_into_tensor(self.betas, t, z.shape)
        alphas_t = 1. - betas_t
        sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z.shape)
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(alphas_t)
        
        # 3. Calculate the expected mean (subtracting the scaled predicted noise)
        model_mean = sqrt_recip_alphas_t * (z - betas_t * model_out / sqrt_one_minus_alphas_cumprod_t)
        
        # 4. In the last step (t=0), return the pure clean signal. 
        # If t > 0, add controlled variance (Langevin dynamics).
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract_into_tensor(self.posterior_variance, t, z.shape)
            noise = torch.randn_like(z)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, verbose=True):
        """Executes full iterative reverse sampling loop from noise to latent space.

        Parameters
        ----------
        shape : tuple
            Target shape for output latent tensor.
        verbose : bool, optional
            Whether to display progress bar. Default is True.

        Returns
        -------
        torch.Tensor
            Denoised latent sequence tensor.
        """
        
        device = self.device
        b = shape[0]
        
        # Start with pure Gaussian noise in the latent space
        z = torch.randn(shape, device=device)
        
        # Iterate backwards, from T-1 (e.g., 999) to 0
        iterator = reversed(range(0, self.num_timesteps))
        if verbose:
            iterator = tqdm(iterator, desc='Denoising Steps', total=self.num_timesteps)
            
        for i in iterator:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z = self.p_sample(z, t, i)
            
        return z

    @torch.no_grad()
    def sample(self, batch_size=10, use_ema=True, verbose=True):
        """Generates synthetic multi-channel physical sensor (HAR) signals from scratch.

        Parameters
        ----------
        batch_size : int, optional
            Number of synthetic sequences to synthesize. Default is 10.
        use_ema : bool, optional
            Whether to use EMA weights during generation. Default is True.
        verbose : bool, optional
            Whether to show progress bar. Default is True.

        Returns
        -------
        torch.Tensor
            Generated signal batch of shape (batch_size, out_channels, original_length).
        """
        # Latent tensor shape: [Batch, Latent Channels (4), Latent Length (16)]
        shape = (batch_size, self.channels, self.sequence_length)
        
        # Use ema_scope to generate with stabilized weights
        if use_ema and self.use_ema:
            with self.ema_scope():
                z_samples = self.p_sample_loop(shape, verbose=verbose)
        else:
            z_samples = self.p_sample_loop(shape, verbose=verbose)
            
        # Decode the latent space [Batch, 4, 16] back to physical space [Batch, 6, 64]
        x_samples = self.decode_first_stage(z_samples)
        
        return x_samples
    
    def simple_forward(self, x, target_time_step, block=None):
        """Passes latent encoding through internal embedding layers of UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor.
        target_time_step : int
            Target diffusion timestep.
        block : Any, optional
            Optional intermediate layer block specifier.

        Returns
        -------
        torch.Tensor
            Extracted activation embedding.
        """
        
        timesteps = torch.full(
            (x.size(0),),
            target_time_step,
            device=x.device,
            dtype=torch.long
        )
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior)
        return self.model.forward_emb(z, timesteps, block=block)
    
    def full_forward(self, x, target_time_step):
        """Full forward pass reproducing complete single-step denoising operation
        and decoding result back to physical signal space.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input sample at diffusion timestep $t$. Shape: (B, C, L).
        target_time_step : int
            Diffusion timestep $t$ from which denoising occurs.

        Returns
        -------
        torch.Tensor
            Denoised output signal mapped back to physical space.
        """
        t = torch.full(
            (x.size(0),),
            target_time_step,
            device=x.device,
            dtype=torch.long
        )
        
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior)
        
        pred_noise = self.model(z, t)
        
        betas_t = extract_into_tensor(self.betas, t, z.shape)
        alphas_t = 1. - betas_t
        sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z.shape)
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(alphas_t)
        
        # 3. Calculate the expected mean (subtracting the scaled predicted noise)
        model_mean = sqrt_recip_alphas_t * (z - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # 4. In the last step (t=0), return the pure clean signal. 
        # If t > 0, add controlled variance (Langevin dynamics).
        if target_time_step == 0:
            model_out = model_mean
        else:
            posterior_variance_t = extract_into_tensor(self.posterior_variance, t, z.shape)
            noise = torch.randn_like(z)
            model_out = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        denoised = self.decode_first_stage(model_out)
        return denoised
    
    def get_init_config(self):
        """Returns initialization configuration dictionary for serialization.

        Returns
        -------
        dict
            Dictionary containing model initialization parameters.
        """
        return {
            "unet_model": UNetModel1d(**self.model.get_init_config()),
            "first_stage_model": self.first_stage_model,
            "timesteps": self.num_timesteps,
            "loss_type": self.loss_type,
            "sequence_length": self.sequence_length,
            "channels": self.channels,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "unconditional": self.unconditional,
            "beta_schedule": self.beta_schedule,
            "scale_factor": self.scale_factor,
            "scale_by_std": self.scale_by_std
        }