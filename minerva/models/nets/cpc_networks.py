from torch import nn
import lightning as L
import torch
from minerva.models.nets.time_series.resnet import _ResNet1D

# CNN encoder for CPC for HAR


class CNN(L.LightningModule):

    def __init__(self):
        """
        Convolutional Neural Network (CNN) encoder for CPC (Contrastive Predictive Coding)
        for Human Activity Recognition (HAR).

        This class serves as a wrapper for the Convolutional1DEncoder class,
        providing an easy-to-use interface for the CPC model.
        """
        super(CNN, self).__init__()
        self.encoder = Convolutional1DEncoder()

    def forward(self, x):
        return self.encoder(x)


class Convolutional1DEncoder(L.LightningModule):

    def __init__(self, input_size=6, kernel_size=3, stride=1, padding=1):
        """
        1D Convolutional Encoder for CPC.

        This encoder consists of a sequence of convolutional blocks that process the
        input time series data.

        Parameters
        ----------
        input_size : int, optional
            Number of input channels, by default 6.
        kernel_size : int, optional
            Size of the convolutional kernel, by default 3.
        stride : int, optional
            Stride of the convolution, by default 1.
        padding : int, optional
            Padding for the convolution, by default 1.
        """
        super(Convolutional1DEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(
                input_size,
                32,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
            ),
            ConvBlock(
                32,
                64,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
            ),
            ConvBlock(
                64,
                128,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
            ),
        )

    def forward(self, x):
        # print("x shape: ", x.shape)
        encoder = self.encoder(x)
        encoder = encoder.permute(0, 2, 1)
        return encoder


class ResNetEncoder(_ResNet1D):
    def __init__(self, permute=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permute = permute

    def forward(self, x):
        x = super().forward(x)

        if self.permute:
            x = x.permute(0, 2, 1).contiguous()

        return x


class ConvBlock(L.LightningModule):

    def __init__(
        self,
        in_channels=6,
        out_channels=128,
        kernel_size=1,
        stride=1,
        padding=1,
        padding_mode="reflect",
        dropout_prob=0.2,
    ):
        """
        Convolutional Block for the 1D Convolutional Encoder.

        This block consists of a convolutional layer followed by a ReLU activation and dropout.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels, by default 6.
        out_channels : int, optional
            Number of output channels, by default 128.
        kernel_size : int, optional
            Size of the convolutional kernel, by default 1.
        stride : int, optional
            Stride of the convolution, by default 1.
        padding : int, optional
            Padding for the convolution, by default 1.
        padding_mode : str, optional
            Padding mode for the convolution, by default 'reflect'.
        dropout_prob : float, optional
            Dropout probability, by default 0.2.
        """
        super(ConvBlock, self).__init__()

        # 1D convolutional layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, inputs):
        conv = self.conv(inputs)
        relu = self.relu(conv)
        dropout = self.dropout(relu)

        return dropout


# ProjectionHead for CPC for HAR


class PredictionNetwork(L.LightningModule):

    def __init__(self, in_channels=256, out_channels=128):
        """
        Projection head for CPC used in Human Activity Recognition (HAR).

        This network projects the encoded representations to a lower-dimensional space
        to facilitate the contrastive learning process.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels, by default 256.
        out_channels : int, optional
            Number of output channels, by default 128.
        """
        super(PredictionNetwork, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        prediction = self.Wk(x)
        return prediction


# Autoregressive model for CPC for HAR


class HARCPCAutoregressive(L.LightningModule):

    def __init__(
        self,
        input_size=128,
        hidden_size=256,
        num_layers=2,
        bidirectional=False,
        batch_first=True,
        dropout=0.2,
    ):
        """
        Autoregressive model for CPC used in Human Activity Recognition (HAR).

        This network models the temporal dependencies in the feature space.

        Parameters
        ----------
        input_size : int, optional
            Number of input features, by default 128.
        hidden_size : int, optional
            Number of hidden units, by default 256.
        num_layers : int, optional
            Number of recurrent layers, by default 2.
        bidirectional : bool, optional
            If True, becomes a bidirectional GRU, by default False.
        batch_first : bool, optional
            If True, the input and output tensors are provided as (batch, seq, feature), by default True.
        dropout : float, optional
            Dropout probability, by default 0.2.
        """
        super(HARCPCAutoregressive, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden)
        # print("output shape: ", output.shape)
        return output, hidden


# Combination of the GENC and GAR networks, backbone of the CPC.
class Genc_Gar(torch.nn.Module):

    def __init__(self, g_enc: torch.nn.Module, g_ar: torch.nn.Module):
        """
        Combination of the GENC (encoder) and GAR (autoregressive) networks,
        forming the backbone of the CPC model for HAR.

        Parameters
        ----------
        g_enc: torch.nn.Module
            Encoder network to extract features from the input data.
        g_ar : torch.nn.Module
            Autoregressive network to model temporal dependencies in the feature space.
        """
        super(Genc_Gar, self).__init__()
        self.g_enc = g_enc
        self.g_ar = g_ar

    def forward(self, x):
        x = self.g_enc(x)
        x, _ = self.g_ar(x, None)
        x = x[:, -1, :]
        return x


# Prediction Head


class HARPredictionHead(L.LightningModule):

    def __init__(self, num_classes: int = 9):
        """
        Prediction head for Human Activity Recognition (HAR).

        This network takes the encoded and temporally modeled features and outputs the final
        activity classification.

        Parameters
        ----------
        num_classes : int, optional
            Number of activity classes to predict, by default 9 (RW_waist).
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# Linear Classifier
class LinearClassifier(L.LightningModule):

    def __init__(
        self,
        backbone: L.LightningModule,
        head: L.LightningModule,
        num_classes: int = 6,
        learning_rate: float = 0.001,
        flatten: bool = True,
        freeze_backbone: bool = False,
        loss_fn: torch.nn.modules.loss._Loss = None,
    ):
        """
        A linear classifier model built on top of a backbone and a head network, designed for tasks
        such as classification. This model leverages PyTorch Lightning for easier training and
        evaluation.

        Parameters
        ----------
        backbone : L.LightningModule
            The backbone network used for feature extraction.
        head : L.LightningModule
            The head network used for the final classification.
        num_classes : int, optional
            The number of output classes, by default 6.
        learning_rate : float, optional
            The learning rate for the optimizer, by default 0.001.
        flatten : bool, optional
            Whether to flatten the output of the backbone before passing it to the head, by default True.
        freeze_backbone : bool, optional
            Whether to freeze the backbone during training, by default False.
        loss_fn : torch.nn.modules.loss._Loss, optional
            The loss function to use, by default CrossEntropyLoss.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.flatten = flatten
        self.loss_fn = loss_fn
        self.freeze_backbone = freeze_backbone

        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def calculate_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, stage_name: str
    ) -> dict:
        """Calculate metrics for the given batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted labels.
        y_true : torch.Tensor
            True labels.

        Returns
        -------
        dict
            Dictionary of metrics.
        """
        assert stage_name in [
            "train",
            "val",
            "test",
        ], f"Invalid stage name: {stage_name}"

        # Our metrics dictionary
        metrics = dict()

        # Move to CPU and detach
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()

        # Calculate accuracy
        y_pred = torch.argmax(y_pred, dim=1)
        acc = float((y_pred == y_true).float().mean())
        metrics = {f"{stage_name}_accuracy": acc}

        # Add more metrics if wanted...., e.g. f1, precision, recall, etc.
        # ...

        return metrics

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
        return self.head(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # Unpack
        x, y = batch
        # Forward pass
        logits = self.forward(x)
        # Calculate loss
        loss = self.loss_fn(logits, y)
        # Log loss
        self.log(f"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # return a dictionary of metrics (loss must be present)
        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # Unpack
        x, y = batch
        # Forward pass
        logits = self.forward(x)
        # Calculate loss
        loss = self.loss_fn(logits, y)
        # Log loss
        self.log(f"val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # calculate metrics and get a dictionary of metrics and log all metrics
        metrics = self.calculate_metrics(logits, y, stage_name="val")
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        # return a dictionary of metrics (loss must be present)
        metrics["loss"] = loss
        return metrics

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        # Unpack
        x, y = batch
        # Forward pass
        logits = self.forward(x)
        # Calculate loss
        loss = self.loss_fn(logits, y)
        # Log loss
        self.log(f"test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # calculate metrics and get a dictionary of metrics and log all metrics
        metrics = self.calculate_metrics(logits, y, stage_name="test")
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        # return a dictionary of metrics (loss must be present)
        metrics["loss"] = loss
        metrics["y_true"] = y
        metrics["y_pred"] = logits
        return metrics

    def _freeze(self, model):
        """Freezes the model, i.e. sets the requires_grad parameter of all the
        parameters to False.

        Parameters
        ----------
        model : type
            The model to freeze
        """
        for param in model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        """Configures the optimizer. If `update_backbone` is True, it will
        update the parameters of the backbone and the head. Otherwise, it will
        only update the parameters of the head.
        """
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
