import torch
from torch import nn
import lightning as pl
from typing import List, Tuple
from minerva.transforms.tfc import TFC_Transforms
from minerva.models.nets.tfc import TFC_Backbone, TFC_PredicionHead


class TFC_Model(pl.LightningModule):
    def __init__(self, input_channels, TS_length, num_classes, single_encoding_size, backbone = None, pred_head = True, loss = None, learning_rate = 3e-4, transform = None):
        super(TFC_Model, self).__init__()
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = TFC_Backbone(input_channels, TS_length, single_encoding_size = single_encoding_size)
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=3e-4)