# /workspaces/Minerva-Dev/minerva/models/ssl/simclr.py

import torch
import torch.nn as nn
from torch.nn.functional import normalize
import lightning as L


class MLPHead(nn.Module):
    """ A simple multilayer perceptron (MLP) head for SimCLR.
    This class `MLPHead` defines a simple neural network module with 
    a sequence of two linear layers and a batch normalization layer,
    implemented using PyTorch. Here's what each method does:

    * `__init__`: Initializes the module with the specified input, hidden,
    and output dimensions, and defines the sequence of layers.
    * `forward`: Defines the forward pass through the network, where the
    input `x` is passed through the sequence of layers defined in `__init__`.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class LinearEvalHead(nn.Module):
    """ A simple linear evaluation head for SimCLR.
    This class `LinearEvalHead` defines a simple linear evaluation head for SimCLR,
    implemented using PyTorch. Here's what each method does:

    * `__init__`: Initializes the module with the specified input and output dimensions,
    and defines the linear layer.
    * `forward`: Defines the forward pass through the linear layer, where the
    input `x` is passed through the linear layer defined in `__init__`.
    """
    def __init__(self, input_dim, num_classes):
        super(LinearEvalHead, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

class SimCLR(L.LightningModule):
    """ A SimCLR model for self-supervised learning.
    This class `SimCLR` defines a SimCLR model for self-supervised learning, 
    implemented using PyTorch. Here's what each method does:

    * `__init__`: Initializes the module with the specified backbone, projector,
    hidden, output, temperature, learning rate, test metric, and number of classes.
    * `forward`: Defines the forward pass through the backbone and projector,
    where the input `x` is passed through the backbone and projector defined in `__init__`.
    * `nt_xent_loss`: Computes the negative training cross-entropy loss for SimCLR,
    where the input `projections` is passed through the negative training cross-entropy loss function.

    * `training_step`: Defines the training step for SimCLR, where the input `batch`
    is passed through the backbone and projector, and the negative training cross-entropy loss is computed.
    * `test_step`: Defines the test step for SimCLR, where the input `batch` is passed through the backbone and projector,
    and the negative training cross-entropy loss is computed.
    * `validation_step`: Defines the validation step for SimCLR, where the input `batch` is passed through the backbone and projector,
    and the negative training cross-entropy loss is computed.
    * `predict_step`: Defines the predict step for SimCLR, where the input `batch` is passed through the backbone and projector.
    * `configure_optimizers`: Configures the optimizer for SimCLR, where the learning rate is defined in `__init__`.

    References:
    Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations.
    In International Conference on machine learning (pp. 1597-1607). PMLR.
    - SimCLR: https://arxiv.org/abs/2002.05709
    """
    def __init__(self, backbone, projector_dim, hidden_dim, output_dim, temperature=0.5, lr=1e-3, test_metric=None, num_classes=None):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = MLPHead(projector_dim, hidden_dim, output_dim)
        self.temperature = temperature
        self.lr = lr
        self.test_metric = test_metric
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = LinearEvalHead(output_dim, num_classes)  # Add linear classifier

    
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        projections = self.projector(flattened)
        return normalize(projections, dim=1)

    def nt_xent_loss(self, projections):
        batch_size = projections.size(0)
        similarity_matrix = torch.mm(projections, projections.T) / self.temperature
        labels = torch.arange(batch_size).to(projections.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss

    def training_step(self, batch, batch_idx):
        images, _ = batch  # Assuming labels are not used for unsupervised learning
        projections = self(images)
        loss = self.nt_xent_loss(projections)
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        projections = self(images)
        loss = self.nt_xent_loss(projections)
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        projetions = self(images)
        loss = self.nt_xent_loss(projetions)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels = batch
        projections = self(images)
        return projections
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
