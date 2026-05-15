import lightning as L
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from torch import nn
from typing import Optional, Callable
import torch
from minerva.schedulers.warmup_cosine_annealing import WarmupCosineAnnealingLR


class DIET(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        linear_head: Optional[torch.nn.Module] = None,
        num_data: Optional[int] = None,
        flatten: bool = True,
        adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        loss: Callable = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 3e-4,
        wca_scheduler_total_epochs: Optional[int] = None,
    ):
        """
        DIET model.

        Parameters
        ----------
        backbone : torch.nn.Module
            Backbone model.
        linear_head: torch.nn.Module, optional
            Linear head that computes logits from embeddings of the data input, by default None.
            If None, the linear head is automatically defined before training. The lengths of
            both training dataset and linear head output must match.
        num_data : int, optional
            Total number of samples in the training dataset, by default None. If None, the length
            of the training dataset is computed before the training in the setup() function.
        flatten : bool
            If True, the output of the backbone is flattened before the linear layer,
            by default True.
        adapter : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
            If not None, an adapter is added after the backbone and before the flatten process,
            by default None.
        loss : Callable
            Loss function, by default CrossEntropyLoss with label smoothing 0.8.
        learning_rate : float, optional
            Learning rate used in the optimizer, by default 3e-4.
        weight_decay : float, optional
            Weight decay used in the optimizer, by default 3e-4.
        wca_scheduler_total_epochs : int, optional
            Total number of epochs for the WarmupCosineAnnealing scheduler, by default None.
            Must be None or an integer greater than 10. If None, no scheduler is used.
        """
        super(DIET, self).__init__()
        # Defining layers
        self.backbone = backbone
        self.linear_head = linear_head
        self.num_data = num_data
        # Defining adapter
        self.adapter = adapter
        self.flatten = flatten
        # Defining loss
        self.loss = loss or CrossEntropyLoss(label_smoothing=0.8)
        # Defining other hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.wca_scheduler_total_epochs = wca_scheduler_total_epochs

        if (
            self.wca_scheduler_total_epochs is not None
            and self.wca_scheduler_total_epochs <= 10
        ):
            raise ValueError(
                "Total number of epochs for the WarmupCosineAnnealing scheduler must be greater than 10."
            )

    def setup(self, stage):
        """
        Setup function. If the model lacks a linear head, this function computes the length
        of the training dataset, the encoding size, and creates a linear head accordingly. Also
        checks whether the linear head output matches the length of the training dataset,
        raising an error in case of mismatch.
        """
        if stage != "fit":
            return
        # Get the training dataset
        training_dataset = self.trainer.datamodule.train_dataloader().dataset
        # Update num_data if None
        if self.num_data is None:
            self.num_data = len(training_dataset)
        # Define a linear head if None
        if self.linear_head is None:
            # Simulated input for encoding_size calculation
            random_input = torch.rand(training_dataset[:5][0].shape)
            # Compute the encoding size
            with torch.no_grad():
                # Obtain the embeddings from the random data
                out = self.backbone(random_input)
                if self.adapter:
                    out = self.adapter(out)
                if self.flatten:
                    out = out.flatten(start_dim=1)
            # Computes the encoding size
            encoding_size = out.size(1)
            # Defines the linear head
            self.linear_head = nn.Linear(encoding_size, self.num_data)
        else:
            # Check if the linear head provided matches the length of the training dataset
            assert (
                self.num_data == self.linear_head.out_features
            ), f"Number of samples({self.num_data}) and output of linear head({self.linear_head.out_features}) do not match."

    def forward(self, x):
        x = self.backbone(x)
        if self.adapter:
            x = self.adapter(x)
        if self.flatten:
            x = x.flatten(start_dim=1)
        x = self.linear_head(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        A simple training step.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
        )
        # If self.wca_scheduler_total_epochs is not None, we return the optimizer and the scheduler
        if self.wca_scheduler_total_epochs:
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=10,
                total_epochs=self.wca_scheduler_total_epochs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "train_loss",
                },
            }
        # If self.wca_scheduler_total_epochs is None, we return only the optimizer
        return optimizer