import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Any, Sequence


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
    flatten : bool, optional, default=True
        Whether to flatten the output of the backbone model, by default True
    avg_pooling: bool, optional, default=True
        Whether to apply a global average pooling to the backbone output.
        If flatten is True it will be applied before flattening, by default True
    k : int, optional
        Number of target views to use, it will k+1 total views, make sure your Dataset class yields k+1 views, by default 3
    lr : float, optional
        Learning rate for the optimizer, by default 0.125
    loss_fn : Callable, optional
        Loss function used for training. Default is cosine similarity loss.
        This loss will be multiplied by -1 when computing total loss
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_dim: int = 2048,
        hid_dim: int = 2048,
        out_dim: int = 2048,
        flatten: bool = True,
        avg_pooling: bool = True,
        k: int = 3,
        lr: float = 0.125,
        loss_fn: Optional[nn.Module] = None,
    ):
        super(FastSiam, self).__init__()
        self.k = k
        self.lr = lr
        self.out_dim = out_dim
        self.flatten = flatten

        self.backbone = backbone
        self.projection_head = SimSiamMLPHead(
            [in_dim, 512, out_dim],
            activation_cls=nn.ReLU,
            batch_norm=True,
            final_bn=False,
            final_relu=False,
        )
        self.prediction_head = SimSiamMLPHead(
            [out_dim, hid_dim, out_dim],
            activation_cls=nn.ReLU,
            batch_norm=True,
            final_bn=False,
            final_relu=False,
        )
        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

        self.global_avg_pool = None
        if avg_pooling:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        f = self.backbone(x)
        if self.global_avg_pool:
            f = self.global_avg_pool(f)
        if self.flatten:
            f = f.flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def _single_step_arbitrary_k(self, batch: Any):
        # borrowed from lightly
        # source: https://github.com/lightly-ai/lightly/blob/f5a9f19ac983d88f058030d630761432dbbd98a3/examples/pytorch_lightning/fastsiam.py
        features = [self.forward(view) for view in batch]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(self.k + 1):
            mask = torch.arange(self.k + 1, device=self.device) != i
            loss += -self.loss_fn(ps[i], torch.sum(zs[mask], dim=0) / self.k).mean()
        return loss / (self.k + 1)

    def _single_step(self, batch: Any, log_prefix: str) -> torch.Tensor:
        if len(batch) != self.k + 1:
            raise RuntimeError(
                f"expected {self.k + 1} views, but got {len(batch)}, is your Dataset class yielding k+1 views?"
            )

        # since the user will probably use k = 3 most of the time
        # lets just inline the loss here
        if self.k == 3:
            z0, p0 = self.forward(batch[0])
            z1, p1 = self.forward(batch[1])
            z2, p2 = self.forward(batch[2])
            z3, p3 = self.forward(batch[3])

            d1 = self.loss_fn(p0, (z1 + z2 + z3) / 3).mean()
            d2 = self.loss_fn(p1, (z0 + z2 + z3) / 3).mean()
            d3 = self.loss_fn(p2, (z0 + z1 + z3) / 3).mean()
            d4 = self.loss_fn(p3, (z0 + z1 + z2) / 3).mean()

            loss = (-1 / 4) * (d1 + d2 + d3 + d4)
        else:
            loss = self._single_step_arbitrary_k(batch)

        self.log(
            f"{log_prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._single_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
