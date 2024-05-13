import warnings
from enum import Enum
from typing import Optional, Tuple

import lightning as L
import torch
from torch import nn
from torchmetrics import JaccardIndex

from minerva.models.nets.vit import _VisionTransformerBackbone
from minerva.utils.upsample import Upsample, resize


class _SETRUPHead(nn.Module):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

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
        """
        Initializes the SETR model.

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
    """Multi level feature aggretation head of SETR.

    MLA head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.
    """

    def __init__(
        self,
        channels: int,
        conv_norm: Optional[nn.Module],
        conv_act: Optional[nn.Module],
        in_channels: list[int],
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
        image_size: int | tuple[int, int],
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
    ):
        """
        Initializes the SETR PUP model.

        Parameters
        ----------
        image_size : int or tuple[int, int]
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
            The dimension of the feed-forward network in the transformer encoder.
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
            The normalization layer to be used in the decoder convolutional layers.
        conv_act : nn.Module
            The activation function to be used in the decoder convolutional layers.
        align_corners : bool
            Whether to align corners during upsampling.

        """
        super().__init__()
        self.encoder = _VisionTransformerBackbone(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            dropout=encoder_dropout,
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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        # x_aux1 = self.aux_head1(x)
        # x_aux2 = self.aux_head2(x)
        # x_aux3 = self.aux_head3(x)
        x = self.decoder(x)
        return x, torch.zeros(1), torch.zeros(1), torch.zeros(1)


class MetricTypeSetR(Enum):
    """Type of metric to be used for evaluation.

    Type of metric to be used for evaluation in the `SETR` model.
    """

    mIoU = "mIoU"

    def get_metric(self, **kwargs) -> nn.Module:
        """Get the metric object.

        Returns
        -------
        nn.Module
            The metric object.
        """

        if self == MetricTypeSetR.mIoU:
            task = kwargs.get("task", "segmentation")
            num_classes = kwargs.get("num_classes")
            avrege = kwargs.get("average", "macro")
            return JaccardIndex(task=task, num_classes=num_classes, average=avrege)
        raise ValueError(f"Invalid metric type: {self}")


class SETR_PUP(L.LightningModule):

    def __init__(
        self,
        image_size: int | tuple[int, int] = 512,
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
        log_train_metrics: bool = False,
        train_metrics: Optional[nn.Module] = None,
        log_val_metrics: bool = False,
        val_metrics: Optional[nn.Module] = None,
        log_test_metrics: bool = False,
        test_metrics: Optional[nn.Module] = None,
    ):
        """
        Initializes the SetR model.

        Parameters
        ----------
        image_size : int or tuple[int, int]
            The input image size. Defaults to 512.
        patch_size : int
            The size of each patch. Defaults to 16.
        num_layers : int
            The number of layers in the transformer encoder. Defaults to 24.
        num_heads : int
            The number of attention heads in the transformer encoder. Defaults to 16.
        hidden_dim : int
            The hidden dimension of the transformer encoder. Defaults to 1024.
        mlp_dim : int
            The dimension of the MLP layers in the transformer encoder. Defaults to 4096.
        encoder_dropout : float
            The dropout rate for the transformer encoder. Defaults to 0.1.
        num_classes : int
            The number of output classes. Defaults to 1000.
        norm_layer : nn.Module, optional
            The normalization layer to be used in the decoder. Defaults to None.
        decoder_channels : int
            The number of channels in the decoder. Defaults to 256.
        num_convs : int
            The number of convolutional layers in the decoder. Defaults to 4.
        up_scale : int
            The scale factor for upsampling in the decoder. Defaults to 2.
        kernel_size : int
            The kernel size for convolutional layers in the decoder. Defaults to 3.
        align_corners : bool
            Whether to align corners during interpolation in the decoder. Defaults to False.
        decoder_dropout : float
            The dropout rate for the decoder. Defaults to 0.1.
        conv_norm : nn.Module, optional
            The normalization layer to be used in the convolutional layers of the decoder. Defaults to None.
        conv_act : nn.Module, optional
            The activation function to be used in the convolutional layers of the decoder. Defaults to None.
        interpolate_mode : str
            The interpolation mode for upsampling in the decoder. Defaults to "bilinear".
        loss_fn : nn.Module, optional
            The loss function to be used during training. Defaults to None.
        log_metrics : bool
            Whether to log metrics during training. Defaults to True.
        metrics : list[MetricTypeSetR], optional
            The metrics to be used for evaluation. Defaults to [MetricTypeSetR.mIoU, MetricTypeSetR.mIoU, MetricTypeSetR.mIoU].

        """
        super().__init__()
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        norm_layer = norm_layer if norm_layer is not None else nn.LayerNorm(hidden_dim)
        conv_norm = (
            conv_norm if conv_norm is not None else nn.SyncBatchNorm(decoder_channels)
        )
        conv_act = conv_act if conv_act is not None else nn.ReLU()
        self.num_classes = num_classes

        self.log_train_metrics = log_train_metrics
        self.log_val_metrics = log_val_metrics
        self.log_test_metrics = log_test_metrics

        if log_train_metrics:
            assert (
                train_metrics is not None
            ), "train_metrics must be provided if log_train_metrics is True"
            self.train_metrics = train_metrics

        if log_val_metrics:
            assert (
                val_metrics is not None
            ), "val_metrics must be provided if log_val_metrics is True"
            self.val_metrics = val_metrics

        if log_test_metrics:
            assert (
                test_metrics is not None
            ), "test_metrics must be provided if log_test_metrics is True"
            self.test_metrics = test_metrics

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
        )

        self.train_step_outputs = []
        self.train_step_labels = []

        self.val_step_outputs = []
        self.val_step_labels = []

        self.test_step_outputs = []
        self.test_step_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss_func(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        y_hat = self.model(x.float())
        loss = self._loss_func(y_hat[0], y.squeeze(1))

        if step_name == "train":
            self.train_step_outputs.append(y_hat[0])
            self.train_step_labels.append(y)
        elif step_name == "val":
            self.val_step_outputs.append(y_hat[0])
            self.val_step_labels.append(y)
        elif step_name == "test":
            self.test_step_outputs.append(y_hat[0])
            self.test_step_labels.append(y)

        self.log_dict(
            {
                f"{step_name}_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def on_train_epoch_end(self):
        if self.log_train_metrics:
            y_hat = torch.cat(self.train_step_outputs)
            y = torch.cat(self.train_step_labels)
            preds = torch.argmax(y_hat, dim=1, keepdim=True)
            self.train_metrics(preds, y)
            mIoU = self.train_metrics.compute()

            self.log_dict(
                {
                    f"train_metrics": mIoU,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.train_step_outputs.clear()
        self.train_step_labels.clear()

    def on_validation_epoch_end(self):
        if self.log_val_metrics:
            y_hat = torch.cat(self.val_step_outputs)
            y = torch.cat(self.val_step_labels)
            preds = torch.argmax(y_hat, dim=1, keepdim=True)
            self.val_metrics(preds, y)
            mIoU = self.val_metrics.compute()

            self.log_dict(
                {
                    f"val_metrics": mIoU,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.val_step_outputs.clear()
        self.val_step_labels.clear()

    def on_test_epoch_end(self):
        if self.log_test_metrics:
            y_hat = torch.cat(self.test_step_outputs)
            y = torch.cat(self.test_step_labels)
            preds = torch.argmax(y_hat, dim=1, keepdim=True)
            self.test_metrics(preds, y)
            mIoU = self.test_metrics.compute()
            self.log_dict(
                {
                    f"test_metrics": mIoU,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.test_step_outputs.clear()
        self.test_step_labels.clear()

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "test")

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int | None = None
    ):
        x, _ = batch
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
