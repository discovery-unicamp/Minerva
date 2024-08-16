import torch
from torch import nn
import lightning as pl
from typing import List, Tuple, Union, Optional
from minerva.transforms.tfc import TFC_Transforms
from minerva.models.nets.tfc import TFC_Conv_Backbone, TFC_PredicionHead
from minerva.losses.ntxent_loss_poly import NTXentLoss_poly
from minerva.transforms.transform import _Transform
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torchmetrics import F1Score, Accuracy
from minerva.models.loaders import LoadableModule


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
        backbone: Union[nn.Module, LoadableModule] = None,
        pred_head: Union[bool, nn.Module] = True,
        loss: _Loss = None,
        learning_rate: float = 3e-4,
        transform: _Transform = None,
        device: str = "cuda",
        batch_size: int = 42,
        pipeline: str = "both",
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
        - backbone: Union[nn.Module, LoadableModule]
            The backbone of the model. If None, a default backbone is created as a convolutional neural network
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
        """
        super(TFC_Model, self).__init__()
        self.num_classes = num_classes
        self.pipeline = pipeline
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = TFC_Conv_Backbone(
                input_channels, TS_length, single_encoding_size=single_encoding_size
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

    def forward(
        self, x_t: torch.Tensor, x_f: torch.Tensor = None, all: bool = False
    ) -> torch.Tensor:  # "all" is useful for validation of acurracy and latent space
        """
        The forward method of the model. It receives the input data in the time domain and frequency domain and returns the prediction of the model.

        Parameters
        ----------
        - x_t: torch.Tensor
            The input data in the time domain
        - x_f: torch.Tensor
            The input data in the frequency domain
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
        if x_f is None:
            x_t, _, x_f, _ = self.transform(x_t)
        h_t, z_t, h_f, z_f = self.backbone(x_t, x_f)
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
            pred = self.forward(data, data_f)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
        else:
            h_t, z_t, h_f, z_f = self.forward(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)
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
            pred = self.forward(data, data_f)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
        else:
            h_t, z_t, h_f, z_f = self.forward(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)
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
            pred = self.forward(data, data_f)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
            f1 = F1Score(task="multiclass", num_classes=self.num_classes)(
                pred.cpu().argmax(dim=1), labels.cpu()
            )
            acc = Accuracy(task="multiclass", num_classes=self.num_classes)(
                pred.cpu().argmax(dim=1), labels.cpu()
            )

        else:
            h_t, z_t, h_f, z_f = self.forward(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)
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

        if f1 is not None and acc is not None:
            self.log("F1-score", f1, prog_bar=True)
            self.log("accuracy", acc, prog_bar=True)
            return loss, f1, acc
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
            pred = self.forward(data, data_f)
            return pred

        else:
            _, z_t, _, z_f = self.forward(data, data_f)
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
