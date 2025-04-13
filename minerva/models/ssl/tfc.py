import torch
from torch import nn
import lightning as pl
from typing import List, Tuple, Union, Optional, Dict
from minerva.transforms.tfc import TFC_Transforms
from minerva.models.nets.tfc import TFC_Backbone, TFC_PredicionHead
from minerva.losses.ntxent_loss_poly import NTXentLoss_poly
from minerva.transforms.transform import _Transform
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torchmetrics import F1Score, Accuracy
from minerva.models.loaders import LoadableModule
from torchmetrics import Metric


class TFC_Model(pl.LightningModule):
    """
    Main class for the Temporal-Frequency Convolutional (TFC) model.
    The model is composed of a backbone and a prediction head. The backbone is (by default) a Convolutional Neural Network (CNN) that extracts features
    from the input data in the time domain and frequency domain. The prediction head is a fully connected layer that classifies the features extracted
    by the backbone in the given classes.
    This class can be trained with a supervised learning approach or with a self-supervised learning approach, or even with single pipelines.
    This class implements the training and validation steps for the model, as well as the forward method and the configuration of the optimizer, as
    requeired by pytorch-lightning.
    """

    def __init__(
        self,
        input_channels: int,
        TS_length: int,
        num_classes: Optional[int] = None,
        single_encoding_size: int = 128,
        backbone: Optional[Union[TFC_Backbone, LoadableModule]] = None,
        pred_head: Union[bool, nn.Module] = True,
        loss: _Loss = None,
        learning_rate: float = 3e-4,
        transform: _Transform = None,
        device: str = "cuda",
        batch_size: int = 42,
        pipeline: str = "both",
        time_encoder: Optional[nn.Module] = None,
        frequency_encoder: Optional[nn.Module] = None,
        time_projector: Optional[nn.Module] = None,
        frequency_projector: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        batch_1_correction=False,
    ):
        """
        The constructor of the TFC_Model class.

        Parameters
        ----------
        - input_channels: int
            The number of channels in the input data
        - TS_length: int
            The number of time steps in the input data
        - num_classes: Optional[int]
            The number of downstream classes in the dataset, if none, the model is trained in a self-supervised learning approach
        - single_encoding_size: int
            The size of the encoding in the latent space of frequency or time domain individually
        - backbone: Optional[Union[TFC_Backbone, LoadableModule]]
            The backbone of the model. If None, a default backbone is created with the encoders and projectors provided. If a LoadableModule is provided, it is used as the backbone.
            If provided, make sure you really know what you are doing.
        - pred_head: Union[bool, nn.Module]
            If True, a prediction head (MLP) is added to the model. If False or None, the model is trained in a self-supervised learning approach. If a nn.Module is provided, it is used as the prediction head
        - loss: _Loss
            The loss function to be used in the training step. If None, the ntxent_poly is used for pretrain or the CrossEntropyLoss is used for downstream
        - learning_rate: float
            The learning rate of the optimizer
        - transform: _Transform
            The transformation to be applied to the input data. If None, a default transformation is applied that includes data augmentation and frequency domain transformation
        - device: str
            The device to be used in the training of the model, default is 'cuda'
        - batch_size: int
            The batch size of the model
        - pipeline: str
            The pipeline to be used in the training of the model. It can be 'both', 'time' or 'freq'. Default is 'both'. If 'both', the model is trained with both time and frequency domain data as default described in tfc paper.
             If 'time', the model is trained with only time domain data. If 'freq', the model is trained with only frequency domain data obtained by fft. At these scenarios, the input data must have the time and frequency domain
             but the prediction head will be used only for the selected one. Also is necesssary to adapt the prediction head input half size of expected (only single_encoding_size instead of single_encoding_size*2)
        - time_encoder: Optional[nn.Module]
            The encoder to be used in the time domain. If None, a default encoder is used. time_encoder can not be passed if backbone is passed.
        - frequency_encoder: Optional[nn.Module]
            The encoder to be used in the frequency domain. If None, a default encoder is used. frequency_encoder can not be passed if backbone is passed.
        - time_projector: Optional[nn.Module]
            The projector to be used in the time domain. If None, a default projector is used. time_projector can not be passed if backbone is passed.
        - frequency_projector: Optional[nn.Module]
            The projector to be used in the frequency domain. If None, a default projector is used. frequency_projector can not be passed if backbone is passed.
        - train_metrics : Dict[str, Metric], optional
            The metrics to be used during training, by default None
        - val_metrics : Dict[str, Metric], optional
            The metrics to be used during validation, by default None
        - test_metrics : Dict[str, Metric], optional
            The metrics to be used during testing, by default None
        - batch_1_correction: bool
            If True, some parts of the architecture are adapted to
            work with batch size 1. Default is False, which means
            that the model is not adapted to work with batch size 1.
        """
        super(TFC_Model, self).__init__()
        self.num_classes = num_classes
        self.pipeline = pipeline

        if test_metrics is None:
            if (
                num_classes
            ):  # If num_classes is not provided, the model is trained in a self-supervised learning approach, so there is no test metrics
                test_metrics = {
                    "f1": F1Score(task="multiclass", num_classes=self.num_classes),
                    "accuracy": Accuracy(
                        task="multiclass", num_classes=self.num_classes
                    ),
                }

        self.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        if backbone:
            self.backbone = backbone
            assert (
                time_encoder is None
                and frequency_encoder is None
                and time_projector is None
                and frequency_projector is None
            ), "If a backbone is provided, the encoders and projectors must be None"
        else:
            self.backbone = TFC_Backbone(
                input_channels,
                TS_length,
                single_encoding_size=single_encoding_size,
                time_encoder=time_encoder,
                frequency_encoder=frequency_encoder,
                time_projector=time_projector,
                frequency_projector=frequency_projector,
                batch_1_correction=batch_1_correction,
            )
        if pred_head and num_classes:
            if pred_head == True:
                conections = 2 if pipeline == "both" else 1
                self.pred_head = TFC_PredicionHead(
                    num_classes=num_classes,
                    single_encoding_size=single_encoding_size,
                    connections=conections,
                )
            else:
                self.pred_head = pred_head
        else:
            self.pred_head = None

        if loss:
            self.loss_fn = loss
        else:
            if self.pred_head:
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                device = (
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if device == "cuda"
                    else torch.device("cpu")
                )
                self.loss_fn = NTXentLoss_poly(device, batch_size, 0.2, True)
        self.learning_rate = learning_rate
        if transform:
            self.transform = transform
        else:
            self.transform = TFC_Transforms()

    def _compute_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, step_name: str
    ) -> Dict[str, torch.Tensor]:
        """Calculate the metrics for the given step.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output data from the forward pass.
        y : torch.Tensor
            The input data/label.
        step_name : str
            Name of the step. It will be used to get the metrics from the
            `self.metrics` attribute.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary with the metrics values.
        """
        if self.metrics[step_name] is None:
            return {}

        return {
            f"{step_name}_{metric_name}": metric.to(self.device)(y_hat, y)
            for metric_name, metric in self.metrics[step_name].items()
        }

    def forward(
        self, x: torch.Tensor, all: bool = False
    ) -> torch.Tensor:  # "all" is useful for validation of acurracy and latent space
        """
        The forward method of the model. It receives the input data in the time domain and frequency domain and returns the prediction of the model.

        Parameters
        ----------
        - x: torch.Tensor
            The input data
        - all: bool
            If True, the method returns the prediction of the model and the features extracted by the backbone. If False, only the prediction is returned

        Returns
        -------
        - torch.Tensor
            If the model has a prediction head and parameter "all" is False, the method returns the prediction of the model, a tensor with the shape (batch_size, num_classes)
        - tuple
            If the model has not a prediction head, the method returns a tuple with the features extracted by the backbone, h_t, z_t, h_f, z_f respectively.
        - tuple
            If the model has a prediction head and parameter "all" is True, the method returns a tuple with the prediction of the model and the features extracted by the backbone, following the format prediction, h_t, z_t, h_f, z_f.

        """
        self.backbone(x)
        h_t, z_t, h_f, z_f = self.backbone.get_representations()
        if self.pred_head:
            if self.pipeline == "both":
                fea_concat = torch.cat((z_t, z_f), dim=1)
            elif self.pipeline == "time":
                fea_concat = z_t
            elif self.pipeline == "freq":
                fea_concat = z_f
            else:
                raise ValueError("Invalid pipeline")
            pred = self.pred_head(fea_concat)
            if all:
                return pred, h_t, z_t, h_f, z_f
            return pred
        else:
            return h_t, z_t, h_f, z_f

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int):
        """
        The training step of the model. It receives a batch of data and returns the loss of the model.

        Parameters
        ----------
        - batch: Tuple[torch.Tensor, torch.Tensor]
            A tuple with the input data and its labels as X, Y
        - batch_index: int
            The index of the batch in the dataset (not used in this method)

        Returns
        -------
        - loss
            The loss of the model in this training step
        """
        x = batch[0]
        x = x.to(self.device)
        labels = batch[1]
        data, aug1, data_f, aug1_f = self.transform(x)
        if self.pred_head:
            pred = self.forward(data)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
            if self.metrics["train"]:
                metrics = self._compute_metrics(pred, labels, "train")
                self.log_dict(metrics, prog_bar=False)
        else:
            h_t, z_t, h_f, z_f = self.forward(data)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1)
            loss_t = self.loss_fn(h_t, h_t_aug)
            loss_f = self.loss_fn(h_f, h_f_aug)
            l_TF = self.loss_fn(z_t, z_f)
            l_1, l_2, l_3 = (
                self.loss_fn(z_t, z_f_aug),
                self.loss_fn(z_t_aug, z_f),
                self.loss_fn(z_t_aug, z_f_aug),
            )
            loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
            lam = 0.2
            loss = lam * (loss_t + loss_f) + (1 - lam) * loss_c

        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        """
        The validation step of the model. It receives a batch of data and returns the loss of the model.

        Parameters
        ----------
        - batch: Tuple[torch.Tensor, torch.Tensor]
            A tuple with the input data and its labels as X, Y
        - batch_index: int
            The index of the batch in the dataset (not used in this method)

        Returns
        -------
        - loss
            The loss of the model in this validation step


        """
        x = batch[0]
        x = x.to(self.device)
        labels = batch[1]
        data, aug1, data_f, aug1_f = self.transform(x)
        if self.pred_head:
            pred = self.forward(data)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
            if self.metrics["val"]:
                metrics = self._compute_metrics(pred, labels, "val")
                self.log_dict(metrics, prog_bar=False)
        else:
            h_t, z_t, h_f, z_f = self.forward(data)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1)
            loss_t = self.loss_fn(h_t, h_t_aug)
            loss_f = self.loss_fn(h_f, h_f_aug)
            l_TF = self.loss_fn(z_t, z_f)
            l_1, l_2, l_3 = (
                self.loss_fn(z_t, z_f_aug),
                self.loss_fn(z_t_aug, z_f),
                self.loss_fn(z_t_aug, z_f_aug),
            )
            loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
            lam = 0.2
            loss = lam * (loss_t + loss_f) + (1 - lam) * loss_c
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        """
        The test step of the model. It receives a batch of data and returns the loss of the model and f1 score and accuracy if a prediction head is provided.

        Parameters
        ----------
        - batch: Tuple[torch.Tensor, torch.Tensor]
            A tuple with the input data and its labels as X, Y
        - batch_index: int
            The index of the batch in the dataset (not used in this method)

        Returns
        -------
        - loss
            The loss of the model in this test step
        can also return f1 score and accuracy if a prediction head is provided:
        - Tuple[loss, f1, accuracy] types: Tuple[torch.Tensor, float, float]


        """
        x = batch[0]
        x = x.to(self.device)
        labels = batch[1]
        data, aug1, data_f, aug1_f = self.transform(x)

        f1, acc = None, None
        if self.pred_head:
            pred = self.forward(data)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)

            metrics = self._compute_metrics(pred, labels, "test")
            self.log_dict(metrics, prog_bar=True)

        else:
            h_t, z_t, h_f, z_f = self.forward(data)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1)
            loss_t = self.loss_fn(h_t, h_t_aug)
            loss_f = self.loss_fn(h_f, h_f_aug)
            l_TF = self.loss_fn(z_t, z_f)
            l_1, l_2, l_3 = (
                self.loss_fn(z_t, z_f_aug),
                self.loss_fn(z_t_aug, z_f),
                self.loss_fn(z_t_aug, z_f_aug),
            )
            loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
            lam = 0.2
            loss = lam * (loss_t + loss_f) + (1 - lam) * loss_c
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        """
        The predict step of the model. It receives a batch of data and returns the torch tensor of probability of prediction or the latent space if a prediction head is not provided.

        Parameters
        ----------
        - batch: Tuple[torch.Tensor, torch.Tensor]
            A tuple with the input data and its labels as X, Y
        - batch_index: int
            The index of the batch in the dataset (not used in this method)

        Returns
        -------
        - loss
            The loss of the model in this test step
        can also return f1 score and accuracy if a prediction head is provided:
        - Tuple[loss, f1, accuracy] types: Tuple[torch.Tensor, float, float]


        """
        x = batch[0]
        x = x.to(self.device)
        data, _, data_f, _ = self.transform(x)

        if self.pred_head:
            pred = self.forward(data)
            return pred

        else:
            _, z_t, _, z_f = self.forward(data)
            z = torch.cat((z_t, z_f), dim=1)
            return z

    def configure_optimizers(self) -> Optimizer:
        """
        Function that configures the optimizer of the model. It returns an Adam optimizer with the learning rate defined in the constructor.

        Returns
        -------
        - Optimizer
            The optimizer of the model

        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=3e-4,
        )
