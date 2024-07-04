import torch
from torch import nn
import lightning as pl
from typing import List, Tuple
from minerva.transforms.tfc import TFC_Transforms
from minerva.models.nets.tfc import TFC_Conv_Backbone, TFC_PredicionHead


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

    def __init__(self, input_channels, TS_length, num_classes, single_encoding_size, backbone = None, pred_head = True, loss = None, learning_rate = 3e-4, transform = None):
        """
        The constructor of the TFC_Model class.

        Parameters
        ----------
        - input_channels: int
            The number of channels in the input data
        - TS_length: int
            The number of time steps in the input data
        - num_classes: int
            The number of downstream classes in the dataset
        - single_encoding_size: int
            The size of the encoding in the latent space of frequency or time domain individually
        - backbone: TFC_Conv_Backbone
            The backbone of the model. If None, a default backbone is created as a convolutional neural network
        - pred_head: bool
            If True, a prediction head (MLP) is added to the model. If False, the model is trained in a self-supervised learning approach
        - loss: torch.nn.Module
            The loss function to be used in the training step. If None, the CrossEntropyLoss is used
        - learning_rate: float
            The learning rate of the optimizer
        - transform: TFC_Transforms
            The transformation to be applied to the input data. If None, a default transformation is applied that includes data augmentation and frequency domain transformation        
        """
        super(TFC_Model, self).__init__()
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = TFC_Conv_Backbone(input_channels, TS_length, single_encoding_size = single_encoding_size)
        if pred_head:
            self.pred_head = TFC_PredicionHead(num_classes=num_classes, single_encoding_size=single_encoding_size)
        else:
            self.pred_head = None

        if loss:
            self.loss_fn = loss
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        if transform:
            self.transform = transform
        else:
            self.transform = TFC_Transforms()
    
    def forward(self, x_t, x_f, all=False): # "all" is useful for validation of acurracy and latent space
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
        h_t, z_t, h_f, z_f = self.backbone(x_t, x_f)
        if self.pred_head:
            fea_concat = torch.cat((z_t, z_f), dim=1)
            pred = self.pred_head(fea_concat)
            if all:
                return pred, h_t, z_t, h_f, z_f
            return pred
        else:
            return h_t, z_t, h_f, z_f

    def training_step(self, batch, batch_index):
        """
        The training step of the model. It receives a batch of data and returns the loss of the model.

        Parameters
        ----------
        - batch: tuple
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
            pred = self.forward(data,data_f)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
        else:
            h_t, z_t, h_f, z_f = self.forward(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)
            loss_t = self.loss_fn(h_t, h_t_aug)
            loss_f = self.loss_fn(h_f, h_f_aug)
            l_TF = self.loss_fn(z_t, z_f)
            l_1, l_2, l_3 = self.loss_fn(z_t, z_f_aug), self.loss_fn(z_t_aug, z_f), self.loss_fn(z_t_aug, z_f_aug)
            loss_c = (1+ l_TF -l_1) + (1+ l_TF -l_2) + (1+ l_TF -l_3)
            lam = 0.2
            loss = lam *(loss_t + loss_f) + (1- lam)*loss_c
        self.log("train_loss", loss, prog_bar=False)
        return loss
        
    def validation_step(self, batch, batch_index):
        """
        The validation step of the model. It receives a batch of data and returns the loss of the model.

        Parameters
        ----------
        - batch: tuple
            A tuple with the input data and its labels as X, Y
        - batch_index: int
            The index of the batch in the dataset (not used in this method)

        Returns
        -------
        - loss
            The loss of the model in this validation step
        
        
        """
        x = batch[0]
        labels = batch[1]
        data, aug1, data_f, aug1_f = self.transform(x)
        if self.pred_head:
            pred = self.forward(data,data_f)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
        else:
            h_t, z_t, h_f, z_f = self.forward(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)
            loss_t = self.loss_fn(h_t, h_t_aug)
            loss_f = self.loss_fn(h_f, h_f_aug)
            l_TF = self.loss_fn(z_t, z_f)
            l_1, l_2, l_3 = self.loss_fn(z_t, z_f_aug), self.loss_fn(z_t_aug, z_f), self.loss_fn(z_t_aug, z_f_aug)
            loss_c = (1+ l_TF -l_1) + (1+ l_TF -l_2) + (1+ l_TF -l_3)
            lam = 0.2
            loss = lam *(loss_t + loss_f) + (1- lam)*loss_c
        self.log("val_loss", loss, prog_bar=True)
        return loss
        

    def configure_optimizers(self):
        """
        Function that configures the optimizer of the model. It returns an Adam optimizer with the learning rate defined in the constructor.

        Returns
        -------
        - torch.optim.Adam
            The optimizer of the model

        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=3e-4)