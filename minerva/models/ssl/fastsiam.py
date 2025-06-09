import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import lightning as L
from PIL import Image
from torchvision import transforms
from typing import Optional, Any, Callable, Sequence


class SimSiamMLPHead(nn.Sequential):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_cls: type = nn.ReLU,
        batch_norm: bool = False,
        final_bn: bool = False,
        final_relu: bool = False,
        *args,
        **kwargs,
    ):
        """
        A modular implementation of a multi-layer perceptron (MLP) head, designed for SimSiam-style architectures.

        Parameters
        ----------
        layer_sizes : Sequence[int]
            Sequence of integers representing the sizes of each layer in the MLP.
            Must have at least two elements (input and output sizes).
        activation_cls : type, optional
            The class of the activation function to use, by default `torch.nn.ReLU`.
            Must be a subclass of `torch.nn.Module`.
        batch_norm : bool, optional
            Whether to include batch normalization after each hidden layer, by default `False`.
        final_bn : bool, optional
            Whether to include a batch normalization layer after the final layer, by default `False`.
        final_relu : bool, optional
            Whether to include a ReLU activation after the final layer, by default `False`.
        *args, **kwargs :
            Additional arguments passed to the activation function.

        Raises
        ------
        AssertionError
            If `layer_sizes` has fewer than two elements or contains non-positive integers.
        AssertionError
            If `activation_cls` is not a subclass of `torch.nn.Module`.

        Examples
        --------
        >>> head = SimSiamMLPHead([2048, 512, 128], batch_norm=True)
        >>> x = torch.randn(32, 2048)  # Batch of 32 samples with input dim 2048
        >>> output = head(x)
        """

        assert (
            len(layer_sizes) >= 2
        ), "Multilayer perceptron must have at least 2 layers"
        assert all(
            isinstance(ls, int) and ls > 0 for ls in layer_sizes
        ), "Layer sizes must be positive integers"
        assert issubclass(
            activation_cls, nn.Module
        ), "activation_cls must inherit from torch.nn.Module"

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            layers.append(activation_cls(*args, **kwargs))

        # Final layer
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        if final_bn:
            layers.append(nn.BatchNorm1d(layer_sizes[-1]))
        if final_relu:
            layers.append(activation_cls(*args, **kwargs))

        super().__init__(*layers)


class FastSiam(L.LightningModule):
    """
    A LightningModule implementation for FastSiam, a self-supervised learning framework.

    Tris approach for self-supervised learning was proposed by Pototzky et al., (2022) [1] in
    "FastSiam: Resource-Efficient Self-supervised Learning on a Single GPU".

    [1] Pototzky, D., Sultan, A., Schmidt-Thieme, L. (2022). FastSiam: Resource-Efficient
    Self-supervised Learning on a Single GPU. In: Andres, B., Bernard, F., Cremers, D.,
    Frintrop, S., Goldlücke, B., Ihrke, I. (eds) Pattern Recognition. DAGM GCPR 2022.
    Lecture Notes in Computer Science, vol 13485. Springer, Cham.
    https://doi.org/10.1007/978-3-031-16788-1_4


    Parameters
    ----------
    backbone : nn.Module
        The backbone neural network for feature extraction (e.g., ResNet).
    in_dim : int, optional
        Input dimension for the projector network, by default 2048.
    hid_dim : int, optional
        Hidden dimension for the projector and predictor networks, by default 512.
    out_dim : int, optional
        Output dimension for the projector and predictor networks, by default 128.
    K : int, optional
        Number of target_branch views to generate, by default 3.
    momentum : float, optional
        Momentum factor for updating the target_branch, by default 0.996.
    lr : float, optional
        Learning rate for the optimizer, by default 1e-3.
    test_metric : Optional[Callable], optional
        A callable to compute the test metric, by default None.
    num_classes : Optional[int], optional
        Number of classes for classification tasks, by default None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_dim: int = 2048,
        hid_dim: int = 2048,
        out_dim: int = 2048,
        K: int = 3,
        momentum: float = 0.996,
        lr: float = 1e-3,
        test_metric: Optional[Callable] = None,
        num_classes: Optional[int] = None,
    ):
        super(FastSiam, self).__init__()
        self.K = K
        self.momentum = momentum
        self.lr = lr
        self.test_metric = test_metric
        self.num_classes = num_classes

        # Prediction branch
        self.backbone = backbone
        self.prediction_branch_projector = SimSiamMLPHead(
            [in_dim, 512, out_dim],
            activation_cls=nn.ReLU,
            batch_norm=True,
            final_bn=False,
            final_relu=False,
        )
        self.prediction_branch_predictor = SimSiamMLPHead(
            [out_dim, hid_dim, out_dim],
            activation_cls=nn.ReLU,
            batch_norm=True,
            final_bn=False,
            final_relu=False,
        )

        # Target branch (momentum updated)
        self.target_branch_backbone = copy.deepcopy(backbone)
        self.target_branch_projector = SimSiamMLPHead(
            [in_dim, hid_dim, out_dim],
            activation_cls=nn.ReLU,
            batch_norm=True,
            final_bn=False,
            final_relu=False,
        )

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    @torch.no_grad()
    def update_target_branch(self) -> None:
        """Momentum update for the target branch."""
        for prediction_branch_param, target_branch_param in zip(
            self.backbone.parameters(), self.target_branch_backbone.parameters()
        ):
            target_branch_param.data = (
                self.momentum * target_branch_param.data
                + (1.0 - self.momentum) * prediction_branch_param.data
            )
        for prediction_branch_param, target_branch_param in zip(
            self.prediction_branch_projector.parameters(),
            self.target_branch_projector.parameters(),
        ):
            target_branch_param.data = (
                self.momentum * target_branch_param.data
                + (1.0 - self.momentum) * prediction_branch_param.data
            )

    def forward(self, views: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the prediction branch and target branches."""
        # Ensure views are in the correct format (BCHW)
        if not isinstance(views, list):
            views = [views]

        # Process prediction branch
        prediction_branch_features = self.backbone(views[-1])  # Use the last view
        prediction_branch_features = self.global_avg_pool(prediction_branch_features)
        prediction_branch_features = torch.flatten(
            prediction_branch_features, start_dim=1
        )
        prediction_branch_projected = self.prediction_branch_projector(
            prediction_branch_features
        )
        prediction_branch_predicted = self.prediction_branch_predictor(
            prediction_branch_projected
        )

        target_branch_projected_list = []
        with torch.no_grad():
            for i in range(self.K):
                # Process target branch
                target_branch_features = self.target_branch_backbone(views[i])
                target_branch_features = self.global_avg_pool(target_branch_features)
                target_branch_features = torch.flatten(
                    target_branch_features, start_dim=1
                )
                target_branch_projected = self.target_branch_projector(
                    target_branch_features
                )
                target_branch_projected_list.append(target_branch_projected)

        avg_target_branch_projected = sum(target_branch_projected_list) / self.K

        return prediction_branch_predicted, avg_target_branch_projected.detach()

    def _single_step(self, batch: Any, K: int, log_prefix: str) -> torch.Tensor:
        """Perform a single training, validation, or test step."""
        images, *_ = batch

        # Generate augmented views
        augmented_views = [
            torch.stack([self.ensure_tensor(img)[0] for img in images]).to(self.device)
            for _ in range(K + 1)
        ]

        # Initialize total loss
        total_loss = 0

        # Loop over all views as the prediction view
        for i in range(K + 1):
            # Prediction branch uses view i
            prediction_branch_features = self.backbone(augmented_views[i])
            prediction_branch_features = self.global_avg_pool(
                prediction_branch_features
            )
            prediction_branch_features = torch.flatten(
                prediction_branch_features, start_dim=1
            )
            prediction_branch_projected = self.prediction_branch_projector(
                prediction_branch_features
            )
            prediction_branch_predicted = self.prediction_branch_predictor(
                prediction_branch_projected
            )

            # Target branch averages over all other views
            target_branch_projected_list = []
            with torch.no_grad():
                for j in range(K + 1):
                    if j != i:
                        target_branch_features = self.target_branch_backbone(
                            augmented_views[j]
                        )
                        target_branch_features = self.global_avg_pool(
                            target_branch_features
                        )
                        target_branch_features = torch.flatten(
                            target_branch_features, start_dim=1
                        )
                        target_branch_projected = self.target_branch_projector(
                            target_branch_features
                        )
                        target_branch_projected_list.append(target_branch_projected)

            avg_target_branch_projected = sum(target_branch_projected_list) / K

            # Normalize loss before summing
            avg_target_branch_projected = F.normalize(
                avg_target_branch_projected, dim=-1, p=2
            )

            # Compute loss for the current view
            loss = self.fastsiam_loss(
                prediction_branch_predicted, avg_target_branch_projected.detach()
            )
            total_loss += loss

        # Average the loss over all views
        total_loss /= K + 1

        # Log the loss
        self.log(
            f"{log_prefix}_loss",
            total_loss,
            batch_size=len(images),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, self.K, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, self.K, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, self.K, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training."""
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]

    def ensure_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Ensure the input image is a PyTorch tensor with the correct format."""
        if not isinstance(image, torch.Tensor):
            # Convert non-tensor input to tensor (assuming numpy array input)
            image = torch.from_numpy(image).float()

        # Ensure the tensor is in BCHW format (batch, channels, height, width)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension if missing

        return image

    @staticmethod
    def fastsiam_loss(
        prediction_branch_pred: torch.Tensor, target_branch_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the FastSiam loss (cosine similarity loss)."""
        prediction_branch_pred = F.normalize(prediction_branch_pred, dim=-1, p=2)
        target_branch_target = F.normalize(target_branch_target, dim=-1, p=2)
        return -(prediction_branch_pred * target_branch_target).sum(dim=-1).mean()
