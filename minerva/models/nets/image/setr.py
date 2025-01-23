import warnings
from typing import Dict, List, Optional, Tuple, Union

import lightning.pytorch as L
import torch
from torch import nn
from torch.optim.adam import Adam
from torchmetrics import Metric

from minerva.engines.engine import _Engine
from minerva.models.nets.image.vit import _VisionTransformerBackbone
from minerva.utils.upsample import Upsample


class _SETRUPHead(nn.Module):
    """Naive upsampling head and Progressive upsampling head of SETR
    (as in https://arxiv.org/pdf/2012.15840.pdf).
    """

    def __init__(
        self,
        channels: int,
        in_channels: int,
        num_classes: int,
        norm_layer: nn.Module,
        conv_norm: nn.Module,
        conv_act: nn.Module,
        num_convs: int,
        up_scale: int,
        kernel_size: int,
        align_corners: bool,
        dropout: float,
        interpolate_mode: str,
    ):
        """The SETR PUP Head.

        Parameters
        ----------
        channels : int
            Number of output channels.
        in_channels : int
            Number of input channels.
        num_classes : int
            Number of output classes.
        norm_layer : nn.Module
            Normalization layer.
        conv_norm : nn.Module
            Convolutional normalization layer.
        conv_act : nn.Module
            Convolutional activation layer.
        num_convs : int
            Number of convolutional layers.
        up_scale : int
            Upsampling scale factor.
        kernel_size : int
            Kernel size for convolutional layers.
        align_corners : bool
            Whether to align corners during upsampling.
        dropout : float
            Dropout rate.
        interpolate_mode : str
            Interpolation mode for upsampling.

        Raises
        ------
        AssertionError
            If kernel_size is not 1 or 3.
        """
        assert kernel_size in [1, 3], "kernel_size must be 1 or 3."

        super().__init__()

        self.num_classes = num_classes
        self.out_channels = channels
        self.cls_seg = nn.Conv2d(channels, self.num_classes, 1)
        self.norm = norm_layer
        conv_norm = conv_norm
        conv_act = conv_act
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 != None else None

        self.up_convs = nn.ModuleList()

        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        self.out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    conv_norm,
                    conv_act,
                    Upsample(
                        scale_factor=up_scale,
                        mode=interpolate_mode,
                        align_corners=align_corners,
                    ),
                )
            )
            in_channels = self.out_channels

    def forward(self, x):
        n, c, h, w = x.shape

        x = x.reshape(n, c, h * w).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)

        if self.dropout is not None:
            x = self.dropout(x)
        out = self.cls_seg(x)

        return out


class _SETRMLAHead(nn.Module):
    """Multi level feature aggretation head of SETR (as in
    https://arxiv.org/pdf/2012.15840.pdf)

    Note: This has not been tested yet!
    """

    def __init__(
        self,
        channels: int,
        conv_norm: Optional[nn.Module],
        conv_act: Optional[nn.Module],
        in_channels: List[int],
        out_channels: int,
        num_classes: int,
        mla_channels: int = 128,
        up_scale: int = 4,
        kernel_size: int = 3,
        align_corners: bool = True,
        dropout: float = 0.1,
        threshold: Optional[float] = None,
    ):
        super().__init__()

        if out_channels is None:
            if num_classes == 2:
                warnings.warn(
                    "For binary segmentation, we suggest using"
                    "`out_channels = 1` to define the output"
                    "channels of segmentor, and use `threshold`"
                    "to convert `seg_logits` into a prediction"
                    "applying a threshold"
                )
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                "out_channels should be equal to num_classes,"
                "except binary segmentation set out_channels == 1 and"
                f"num_classes == 2, but got out_channels={out_channels}"
                f"and num_classes={num_classes}"
            )

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn("threshold is not defined for binary, and defaults to 0.3")

        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold
        conv_norm = (
            conv_norm if conv_norm is not None else nn.SyncBatchNorm(mla_channels)
        )
        conv_act = conv_act if conv_act is not None else nn.ReLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 != None else None
        self.cls_seg = nn.Conv2d(channels, out_channels, 1)

        num_inputs = len(in_channels)

        self.up_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels[i],
                        mla_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    conv_norm,
                    conv_act,
                    nn.Conv2d(
                        mla_channels,
                        mla_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    conv_norm,
                    conv_act,
                    Upsample(
                        scale_factor=up_scale,
                        mode="bilinear",
                        align_corners=align_corners,
                    ),
                )
            )

    def forward(self, x):
        outs = []
        for x, up_conv in zip(x, self.up_convs):
            outs.append(up_conv(x))
        out = torch.cat(outs, dim=1)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.cls_seg(out)
        return out


class _SetR_PUP(nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_convs: int,
        num_classes: int,
        decoder_channels: int,
        up_scale: int,
        encoder_dropout: float,
        kernel_size: int,
        decoder_dropout: float,
        norm_layer: nn.Module,
        interpolate_mode: str,
        conv_norm: nn.Module,
        conv_act: nn.Module,
        align_corners: bool,
        aux_output: bool,
        aux_output_layers: Optional[List[int]],
        original_resolution: Optional[Tuple[int, int]],
    ):
        """Initializes the SETR PUP head.

        Parameters
        ----------
        image_size : int or Tuple[int, int]
            The size of the input image.
        patch_size : int
            The size of each patch in the input image.
        num_layers : int
            The number of layers in the transformer encoder.
        num_heads : int
            The number of attention heads in the transformer encoder.
        hidden_dim : int
            The hidden dimension of the transformer encoder.
        mlp_dim : int
            The dimension of the feed-forward network in the transformer encoder
        num_convs : int
            The number of convolutional layers in the decoder.
        num_classes : int
            The number of output classes.
        decoder_channels : int
            The number of channels in the decoder.
        up_scale : int
            The scale factor for upsampling in the decoder.
        encoder_dropout : float
            The dropout rate for the transformer encoder.
        kernel_size : int
            The kernel size for the convolutional layers in the decoder.
        decoder_dropout : float
            The dropout rate for the decoder.
        norm_layer : nn.Module
            The normalization layer to be used.
        interpolate_mode : str
            The mode for interpolation during upsampling.
        conv_norm : nn.Module
            The normalization layer to be used in the decoder convolutional
            layers.
        conv_act : nn.Module
            The activation function to be used in the decoder convolutional
            layers.
        align_corners : bool
            Whether to align corners during upsampling.
        aux_output: bool
            Whether to use auxiliary outputs. If True, aux_output_layers must
            be provided.
        aux_output_layers: List[int], optional
            The layers to use for auxiliary outputs. Must have exacly 3 values.
        original_resolution: Tuple[int, int], optional
            The original resolution of the input image in the pre-training
            weights. When None, positional embeddings will not be interpolated.

        """
        super().__init__()
        if aux_output:
            assert aux_output_layers is not None, "aux_output_layers must be provided."
            assert (
                len(aux_output_layers) == 3
            ), "aux_output_layers must have 3 values. Only 3 aux heads are supported."

        self.aux_output = aux_output
        self.aux_output_layers = aux_output_layers

        self.encoder = _VisionTransformerBackbone(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            dropout=encoder_dropout,
            aux_output=aux_output,
            aux_output_layers=aux_output_layers,
            original_resolution=original_resolution,
        )

        self.decoder = _SETRUPHead(
            channels=decoder_channels,
            in_channels=hidden_dim,
            num_classes=num_classes,
            num_convs=num_convs,
            up_scale=up_scale,
            kernel_size=kernel_size,
            align_corners=align_corners,
            dropout=decoder_dropout,
            conv_norm=conv_norm,
            conv_act=conv_act,
            interpolate_mode=interpolate_mode,
            norm_layer=norm_layer,
        )

        self.aux_head1 = _SETRUPHead(
            channels=decoder_channels,
            in_channels=hidden_dim,
            num_classes=num_classes,
            num_convs=num_convs,
            up_scale=up_scale,
            kernel_size=kernel_size,
            align_corners=align_corners,
            dropout=decoder_dropout,
            conv_norm=conv_norm,
            conv_act=conv_act,
            interpolate_mode=interpolate_mode,
            norm_layer=norm_layer,
        )

        self.aux_head2 = _SETRUPHead(
            channels=decoder_channels,
            in_channels=hidden_dim,
            num_classes=num_classes,
            num_convs=num_convs,
            up_scale=up_scale,
            kernel_size=kernel_size,
            align_corners=align_corners,
            dropout=decoder_dropout,
            conv_norm=conv_norm,
            conv_act=conv_act,
            interpolate_mode=interpolate_mode,
            norm_layer=norm_layer,
        )

        self.aux_head3 = _SETRUPHead(
            channels=decoder_channels,
            in_channels=hidden_dim,
            num_classes=num_classes,
            num_convs=num_convs,
            up_scale=up_scale,
            kernel_size=kernel_size,
            align_corners=align_corners,
            dropout=decoder_dropout,
            conv_norm=conv_norm,
            conv_act=conv_act,
            interpolate_mode=interpolate_mode,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor):

        if self.aux_output:
            x, aux_results = self.encoder(x)
            x_aux1 = self.aux_head1(aux_results[0])
            x_aux2 = self.aux_head2(aux_results[1])
            x_aux3 = self.aux_head3(aux_results[2])
            x = self.decoder(x)
            return x, x_aux1, x_aux2, x_aux3

        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def load_backbone(self, path: str, freeze: bool = False):
        self.encoder.load_backbone(path)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False


class SETR_PUP(L.LightningModule):
    """SET-R model with PUP head for image segmentation.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the model.
    _compute_metrics(y_hat: torch.Tensor, y: torch.Tensor, step_name: str)
        Compute metrics for the given step.
    _loss_func(y_hat: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], y: torch.Tensor) -> torch.Tensor
        Calculate the loss between the output and the input data.
    _single_step(batch: torch.Tensor, batch_idx: int, step_name: str)
        Perform a single step of the training/validation loop.
    training_step(batch: torch.Tensor, batch_idx: int)
        Perform a single training step.
    validation_step(batch: torch.Tensor, batch_idx: int)
        Perform a single validation step.
    test_step(batch: torch.Tensor, batch_idx: int)
        Perform a single test step.
    predict_step(batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None)
        Perform a single prediction step.
    load_backbone(path: str, freeze: bool = False)
        Load a pre-trained backbone.
    configure_optimizers()
        Configure the optimizer for the model.
    create_from_dict(config: Dict) -> "SETR_PUP"
        Create an instance of SETR_PUP from a configuration dictionary.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 512,
        patch_size: int = 16,
        num_layers: int = 24,
        num_heads: int = 16,
        hidden_dim: int = 1024,
        mlp_dim: int = 4096,
        encoder_dropout: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[nn.Module] = None,
        decoder_channels: int = 256,
        num_convs: int = 4,
        up_scale: int = 2,
        kernel_size: int = 3,
        align_corners: bool = False,
        decoder_dropout: float = 0.1,
        conv_norm: Optional[nn.Module] = None,
        conv_act: Optional[nn.Module] = None,
        interpolate_mode: str = "bilinear",
        loss_fn: Optional[nn.Module] = None,
        optimizer_type: Optional[type] = None,
        optimizer_params: Optional[Dict] = None,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        aux_output: bool = True,
        aux_output_layers: Optional[list[int]] = None,
        aux_weights: Optional[list[float]] = None,
        load_backbone_path: Optional[str] = None,
        freeze_backbone_on_load: bool = True,
        learning_rate: float = 1e-3,
        loss_weights: Optional[list[float]] = None,
        original_resolution: Optional[Tuple[int, int]] = None,
        head_lr_factor: float = 1.0,
        test_engine: Optional[_Engine] = None,
    ):
        """Initialize the SETR model with Progressive Upsampling Head.

        Parameters
        ----------
        image_size : Union[int, Tuple[int, int]], optional
            Size of the input image, by default 512.
        patch_size : int, optional
            Size of the patches to be extracted from the input image, by
            default 16.
        num_layers : int, optional
            Number of transformer layers, by default 24.
        num_heads : int, optional
            Number of attention heads, by default 16.
        hidden_dim : int, optional
            Dimension of the hidden layer, by default 1024.
        mlp_dim : int, optional
            Dimension of the MLP layer, by default 4096.
        encoder_dropout : float, optional
            Dropout rate for the encoder, by default 0.1.
        num_classes : int, optional
            Number of output classes, by default 1000.
        norm_layer : Optional[nn.Module], optional
            Normalization layer, by default None.
        decoder_channels : int, optional
            Number of channels in the decoder, by default 256.
        num_convs : int, optional
            Number of convolutional layers in the decoder, by default 4.
        up_scale : int, optional
            Upscaling factor for the decoder, by default 2.
        kernel_size : int, optional
            Kernel size for the convolutional layers, by default 3.
        align_corners : bool, optional
            Whether to align corners when interpolating, by default False.
        decoder_dropout : float, optional
            Dropout rate for the decoder, by default 0.1.
        conv_norm : Optional[nn.Module], optional
            Normalization layer for the convolutional layers, by default None.
        conv_act : Optional[nn.Module], optional
            Activation function for the convolutional layers, by default None.
        interpolate_mode : str, optional
            Interpolation mode, by default "bilinear".
        loss_fn : Optional[nn.Module], optional
            Loss function, when None defaults to nn.CrossEntropyLoss, by
            default None.
        optimizer_type : Optional[type], optional
            Type of optimizer, by default None.
        optimizer_params : Optional[Dict], optional
            Parameters for the optimizer, by default None.
        train_metrics : Optional[Dict[str, Metric]], optional
            Metrics for training, by default None.
        val_metrics : Optional[Dict[str, Metric]], optional
            Metrics for validation, by default None.
        test_metrics : Optional[Dict[str, Metric]], optional
            Metrics for testing, by default None.
        aux_output : bool, optional
            Whether to use auxiliary outputs, by default True.
        aux_output_layers : list[int], optional
            Layers for auxiliary outputs, when None it defaults to [9, 14, 19].
        aux_weights : list[float], optional
            Weights for auxiliary outputs, when None it defaults [0.3, 0.3, 0.3].
        load_backbone_path : Optional[str], optional
            Path to load the backbone model, by default None.
        freeze_backbone_on_load : bool, optional
            Whether to freeze the backbone model on load, by default True.
        learning_rate : float, optional
            Learning rate, by default 1e-3.
        loss_weights : Optional[list[float]], optional
            Weights for the loss function, by default None.
        original_resolution : Optional[Tuple[int, int]], optional
            The original resolution of the input image in the pre-training
            weights. When None, positional embeddings will not be interpolated.
            Defaults to None.
        head_lr_factor : float, optional
            Learning rate factor for the head. used if you need different
            learning rates for backbone and prediction head, by default 1.0.
        test_engine : Optional[_Engine], optional
            Engine used for test and validation steps. When None, behavior of
            all steps, training, testing and validation is the same, by default None.
        """
        super().__init__()

        if head_lr_factor != 1:
            self.automatic_optimization = False
            self.multiple_optimizers = True
        else:
            self.automatic_optimization = True
            self.multiple_optimizers = False

        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else nn.CrossEntropyLoss(
                weight=(
                    torch.tensor(loss_weights) if loss_weights is not None else None
                )
            )
        )
        norm_layer = norm_layer if norm_layer is not None else nn.LayerNorm(hidden_dim)
        conv_norm = (
            conv_norm if conv_norm is not None else nn.SyncBatchNorm(decoder_channels)
        )
        conv_act = conv_act if conv_act is not None else nn.ReLU()

        if aux_output:
            if aux_output_layers is None:
                aux_output_layers = [9, 14, 19]
                warnings.warn(
                    "aux_output_layers not provided. Using default values [9, 14, 19]."
                )
            if aux_weights is None:
                aux_weights = [0.3, 0.3, 0.3]
                warnings.warn(
                    "aux_weights not provided. Using default values [0.3, 0.3, 0.3]."
                )
            assert (
                len(aux_output_layers) == 3
            ), "aux_output_layers must have 3 values. Only 3 aux heads are supported."

        self.optimizer_type = optimizer_type
        if optimizer_type is not None:
            assert optimizer_params is not None, "optimizer_params must be provided."
            self.optimizer_params = optimizer_params

        self.num_classes = num_classes
        self.aux_weights = aux_weights
        self.head_lr_factor = head_lr_factor

        self.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

        self.model = _SetR_PUP(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            num_convs=num_convs,
            up_scale=up_scale,
            kernel_size=kernel_size,
            conv_norm=conv_norm,
            conv_act=conv_act,
            decoder_channels=decoder_channels,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            norm_layer=norm_layer,
            interpolate_mode=interpolate_mode,
            align_corners=align_corners,
            aux_output=aux_output,
            aux_output_layers=aux_output_layers,
            original_resolution=original_resolution,
        )
        if load_backbone_path is not None:
            self.model.load_backbone(load_backbone_path, freeze_backbone_on_load)

        self.learning_rate = learning_rate
        self.test_engine = test_engine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, step_name: str):
        if self.metrics[step_name] is None:
            return {}

        return {
            f"{step_name}_{metric_name}": metric.to(self.device)(
                torch.argmax(y_hat, dim=1, keepdim=True), y
            )
            for metric_name, metric in self.metrics[step_name].items()
        }

    def _loss_func(
        self,
        y_hat: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss between the output and the input data.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output data from the forward pass.
        y : torch.Tensor
            The input data/label.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        if isinstance(y_hat, tuple):
            y_hat, y_aux1, y_aux2, y_aux3 = y_hat
            loss = self.loss_fn(y_hat, y.long())
            loss_aux1 = self.loss_fn(y_aux1, y.long())
            loss_aux2 = self.loss_fn(y_aux2, y.long())
            loss_aux3 = self.loss_fn(y_aux3, y.long())
            return (
                loss
                + (loss_aux1 * self.aux_weights[0])
                + (loss_aux2 * self.aux_weights[1])
                + (loss_aux3 * self.aux_weights[2])
            )
        loss = self.loss_fn(y_hat, y.long())

        return loss

    def _single_step(self, batch: torch.Tensor, batch_idx: int, step_name: str):
        """Perform a single step of the training/validation loop.

        Parameters
        ----------
        batch : torch.Tensor
            The input data.
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step, either "train" or "val".

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        x, y = batch
        if self.test_engine and (step_name == "test" or step_name == "val"):
            y_hat = self.test_engine(self.model, x)
        else:
            y_hat = self.model(x)

        metrics = self._compute_metrics(y_hat[0], y, step_name)
        loss = self._loss_func(y_hat, y.squeeze(1))

        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        if self.multiple_optimizers:
            optimizers_list = self.optimizers()

            for opt in optimizers_list:
                opt.zero_grad()

            loss = self._single_step(batch, batch_idx, "train")

            self.manual_backward(loss)

            for opt in optimizers_list:
                opt.step()
        else:
            return self._single_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "test")

    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        x, _ = batch
        return self.model(x)[0]

    def load_backbone(self, path: str, freeze: bool = False):
        self.model.load_backbone(path, freeze)

    def configure_optimizers(self):
        if self.multiple_optimizers:
            return (
                [
                    self.optimizer_type(
                        self.model.encoder.parameters(),
                        lr=self.learning_rate,
                        **self.optimizer_params,
                    ),
                    self.optimizer_type(
                        list(self.model.decoder.parameters())
                        + list(self.model.aux_head1.parameters())
                        + list(self.model.aux_head2.parameters())
                        + list(self.model.aux_head3.parameters()),
                        lr=self.learning_rate * self.head_lr_factor,
                        **self.optimizer_params,
                    ),
                ]
                if self.optimizer_type is not None
                else [
                    Adam(self.model.encoder.parameters(), lr=self.learning_rate),
                    Adam(self.model.decoder.parameters(), lr=self.learning_rate),
                ]
            )
        else:
            return (
                self.optimizer_type(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    **self.optimizer_params,
                )
                if self.optimizer_type is not None
                else Adam(self.model.parameters(), lr=self.learning_rate)
            )

    @staticmethod
    def create_from_dict(config: Dict) -> "SETR_PUP":
        return SETR_PUP(**config)
