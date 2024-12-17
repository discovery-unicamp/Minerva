import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import copy
from typing import Tuple, Optional
import lightning as L
from PIL import Image


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):


        """
        Projection head for TriBYOL that maps features to a latent space.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        hidden_dim : int
            Dimensionality of the hidden layer.
        output_dim : int
            Dimensionality of the output projections.
        """

        super(MLPHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TriBYOL(L.LightningModule):
    """
    TriBYOL (Triplet BYOL) for self-supervised learning with a triplet network and triple-view loss.

    Tris approach for self-supervised learning was proposed by Li et al. [1] in 
    "TriBYOL: Triplet BYOL for Self-Supervised Learning".

    [1] Li, G., Togo, R., Ogawa, T., & Haseyama, M. (2022, May). 
    Tribyol: Triplet byol for self-supervised representation learning. 
    In ICASSP 2022-2022 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP) (pp. 3458-3462). IEEE.


    Attributes
    ----------
    online_network : nn.Module
        The backbone network used in the online network.
    online_projector : MLPHead
        The projector for the online network.
    online_predictor : MLPHead
        The predictor for the online network.
    target_network_1 : nn.Module
        The first target network, a copy of the backbone with EMA updates.
    target_network_2 : nn.Module
        The second target network, a copy of the backbone with EMA updates.
    target_projector_1 : MLPHead
        The projector for the first target network.
    target_projector_2 : MLPHead
        The projector for the second target network.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pooling layer.
    tau : float
        Exponential moving average decay rate for updating target networks.
    lr : float
        Learning rate for the optimizer.
    momentum : float
        Momentum for the SGD optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the optimizer.
    enable_augment : bool
        Flag to enable or disable augmentations during training.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 128,
        lr: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 0.0004,
        tau: float = 0.996,
        #enable_augment: bool = True
    ):
        """
        Initializes the TriBYOL model.

        Parameters
        ----------
        backbone : nn.Module
            The backbone network for feature extraction (e.g., ResNet).
        projector_dim : int, optional
            Dimension of the projector input (default is 2048).
        hidden_dim : int, optional
            Dimension of the hidden layer in the MLP projector/predictor (default is 512).
        output_dim : int, optional
            Dimension of the projector output (default is 128).
        lr : float, optional
            Learning rate for the optimizer (default is 0.03).
        momentum : float, optional
            Momentum for the SGD optimizer (default is 0.9).
        weight_decay : float, optional
            Weight decay (L2 regularization) for the optimizer (default is 0.0004).
        tau : float, optional
            Exponential moving average (EMA) decay rate for target networks (default is 0.996).
        enable_augment : bool, optional
            If True, apply data augmentations; otherwise, use plain images (default is True).
        """
        super(TriBYOL, self).__init__()
        self.save_hyperparameters(ignore=['backbone'])

        # Online network
        self.online_network = backbone
        self.online_projector = MLPHead(projector_dim, hidden_dim, output_dim)
        self.online_predictor = MLPHead(output_dim, hidden_dim, output_dim)

        # Target networks
        self.target_network_1 = copy.deepcopy(backbone)
        self.target_network_2 = copy.deepcopy(backbone)
        self.target_projector_1 = MLPHead(projector_dim, hidden_dim, output_dim)
        self.target_projector_2 = MLPHead(projector_dim, hidden_dim, output_dim)

        # EMA decay rate
        self.tau = tau

        # Optimizer hyperparameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        #self.enable_augment = enable_augment

        # Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        """Forward pass for the triplet network."""
        # Online network
        f1 = self.online_network(x1)
        f1 = torch.flatten(self.avgpool(f1), 1)
        p1 = self.online_predictor(self.online_projector(f1))

        # Target network 1
        with torch.no_grad():
            t2 = self.target_projector_1(torch.flatten(self.avgpool(self.target_network_1(x2)), 1))
            t3 = self.target_projector_2(torch.flatten(self.avgpool(self.target_network_2(x3)), 1))

        return p1, t2.detach(), t3.detach()

    def _single_step(self, batch, log_prefix: str):
        """Perform a single step (training or validation)."""
        images, *_ = batch

        # Generate three augmented views
        v1 = torch.stack([self.ensure_tensor(img) for img in images]).to(self.device)
        v2 = torch.stack([self.ensure_tensor(img) for img in images]).to(self.device)
        v3 = torch.stack([self.ensure_tensor(img) for img in images]).to(self.device)

        # Forward pass
        p1, t2, t3 = self(v1, v2, v3)

        # Compute triple-view loss
        loss = self.loss_fn(p1, t2) + self.loss_fn(p1, t3)

        self.log(f"{log_prefix}_loss", loss, batch_size=8, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._single_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._single_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._single_step(batch, "test")

    def configure_optimizers(self):
        """Configure the SGD optimizer."""
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        return optimizer

    def update_target_networks(self):
        """Update target networks using EMA of the online network."""
        for online_params, target_params_1, target_params_2 in zip(
            self.online_network.parameters(),
            self.target_network_1.parameters(),
            self.target_network_2.parameters()
        ):
            target_params_1.data = self.tau * target_params_1.data + (1 - self.tau) * online_params.data
            target_params_2.data = self.tau * target_params_2.data + (1 - self.tau) * online_params.data

    def on_before_optimizer_step(self, optimizer):
        self.update_target_networks()

    def ensure_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Ensure the input image is a PyTorch tensor."""
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
            image = transforms.ToTensor()(image)

        return image

    @staticmethod
    def loss_fn(x: torch.Tensor, y: torch.Tensor):
        """Compute the triple-view loss."""
        x = nn.functional.normalize(x, dim=-1, p=2)
        y = nn.functional.normalize(y, dim=-1, p=2)
        cosine_similarity = (x * y).sum(dim=-1)
        loss = 2 - 2 * cosine_similarity.mean()
        return loss
