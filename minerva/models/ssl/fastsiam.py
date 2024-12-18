import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import lightning as L
from PIL import Image
from torchvision import transforms
from typing import Optional, Any, Callable


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    

class FastSiam(L.LightningModule):
    """
    A LightningModule implementation for FastSiam, a self-supervised learning framework.

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
        Number of teacher views to generate, by default 3.
    momentum : float, optional
        Momentum factor for updating the teacher network, by default 0.996.
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
        hid_dim: int = 512,
        out_dim: int = 128,
        K: int = 3,
        momentum: float = 0.996,
        lr: float = 1e-3,
        test_metric: Optional[Callable] = None,
        num_classes: Optional[int] = None
    ):
        super(FastSiam, self).__init__()
        self.K = K
        self.momentum = momentum
        self.lr = lr
        self.test_metric = test_metric
        self.num_classes = num_classes

        # Student network
        self.backbone = backbone
        self.student_projector = MLPHead(in_dim, hid_dim, out_dim)
        self.student_predictor = MLPHead(out_dim, hid_dim, out_dim)

        # Teacher network (momentum updated)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_projector = MLPHead(in_dim, hid_dim, out_dim)

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze teacher's parameters
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_teacher(self) -> None:
        """Momentum update for the teacher network."""
        for student_param, teacher_param in zip(self.backbone.parameters(), self.teacher_backbone.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + (1.0 - self.momentum) * student_param.data
        for student_param, teacher_param in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + (1.0 - self.momentum) * student_param.data

    def forward(self, views: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the student and teacher networks."""
        # Ensure views are in the correct format (BCHW)
        if not isinstance(views, list):
            views = [views]

        # Process student network
        student_features = self.backbone(views[-1])  # Use the last view
        student_features = self.global_avg_pool(student_features)
        student_features = torch.flatten(student_features, start_dim=1)
        student_projected = self.student_projector(student_features)
        student_predicted = self.student_predictor(student_projected)

        teacher_projected_list = []
        with torch.no_grad():
            for i in range(self.K):
                # Process teacher network
                teacher_features = self.teacher_backbone(views[i])
                teacher_features = self.global_avg_pool(teacher_features)
                teacher_features = torch.flatten(teacher_features, start_dim=1)
                teacher_projected = self.teacher_projector(teacher_features)
                teacher_projected_list.append(teacher_projected)

        avg_teacher_projected = sum(teacher_projected_list) / self.K

        return student_predicted, avg_teacher_projected.detach()

    def _single_step(self, batch: Any, K: int, log_prefix: str) -> torch.Tensor:
        """Perform a single training, validation, or test step."""
        images, *_ = batch

        # Generate augmented views
        augmented_views = [
            torch.stack([self.ensure_tensor(img)[0] for img in images]).to(self.device)
            for _ in range(self.K + 1)
        ]

        # Forward pass
        student_predicted, teacher_target = self(augmented_views)
        loss = self.fastsiam_loss(student_predicted, teacher_target)

        self.log(f"{log_prefix}_loss", loss, batch_size=8, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, self.K, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, self.K, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, self.K, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
    def fastsiam_loss(student_pred: torch.Tensor, teacher_target: torch.Tensor) -> torch.Tensor:
        """Compute the FastSiam loss (cosine similarity loss)."""
        student_pred = F.normalize(student_pred, dim=-1, p=2)
        teacher_target = F.normalize(teacher_target, dim=-1, p=2)
        return -(student_pred * teacher_target).sum(dim=-1).mean()
