import lightning as L
import torch
import torch.nn.functional as F
from minerva.losses.batchwise_barlowtwins_loss import BatchWiseBarlowTwinLoss
from typing import Optional, Callable


class RepeatedModuleList(torch.nn.ModuleList):
    """
    A module list with the same module `cls`, instantiated `size` times.
    """

    def __init__(self, size: int, cls: type, *args, **kwargs):
        """
        Initializes the RepeatedModuleList with multiple instances of a given module class.

        Parameters
        ----------
        size: int
            The number of instances to create.
        cls: type
            The module class to instantiate. Must be a subclass of `torch.nn.Module`.
        *args:
            Positional arguments to pass to the module class constructor.
        **kwargs:
            Keyword arguments to pass to the module class constructor.

        Raises
        ------
        AssertionError:
            If `cls` is not a subclass of `torch.nn.Module`.

        Example
        -------
        >>> class SimpleModule(torch.nn.Module):
        >>>     def __init__(self, in_features, out_features):
        >>>         super().__init__()
        >>>         self.linear = torch.nn.Linear(in_features, out_features)
        >>>
        >>> repeated_modules = RepeatedModuleList(3, SimpleModule, 10, 5)
        >>> print(repeated_modules)
        RepeatedModuleList(
            (0): SimpleModule(
                (linear): Linear(in_features=10, out_features=5, bias=True)
            )
            (1): SimpleModule(
                (linear): Linear(in_features=10, out_features=5, bias=True)
            )
            (2): SimpleModule(
                (linear): Linear(in_features=10, out_features=5, bias=True)
            )
        )
        """

        assert issubclass(
            cls, torch.nn.Module
        ), f"{cls} does not derive from torch.nn.Module"

        super().__init__([cls(*args, **kwargs) for _ in range(size)])


class LearnFromRandomnessModel(L.LightningModule):
    """
    A PyTorch Lightning model for pretraining with the technique
    'Learning From Random Projectors'.

    References
    ----------
    Yi Sui, Tongzi Wu, Jesse C. Cresswell, Ga Wu, George Stein, Xiao Shi Huang, Xiaochen Zhang, Maksims Volkovs.
    "Self-supervised Representation Learning From Random Data Projectors", 2024
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        projectors: torch.nn.ModuleList,
        predictors: torch.nn.ModuleList,
        loss_fn: torch.nn.Module = None,
        adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        flatten: bool = False,
        predictor_training_epochs: Optional[int] = None,
    ):
        """
        Initialize the LFR_Model.

        Parameters
        ----------
        backbone: torch.nn.Module
            The backbone neural network for feature extraction.
        projectors: torch.nn.ModuleList
            A list of projector networks.
        predictors: torch.nn.ModuleList
            A list of predictor networks.
        loss_fn: torch.nn.Module
            The loss function to optimize, by default None. If None, the BatchWiseBarlowTwinLoss is used.
        adapter: Optional[Callable[[torch.Tensor], torch.Tensor]]
            An optional adapter network to be used in the model, by default None.
        learning_rate: float
            The learning rate for the optimizer, by default 1e-3.
        weight_decay: float
            The weight decay for the optimizer, by default 0.0.
        flatten: bool
            Whether to flatten the input tensor or not, by default False.
        predictor_training_epochs: Optional[int]
            The number of epochs to train only the predictors (excluding the backbone), by default None.
            If None, zero, or negative, both the predictors and backbone are trained in every epoch. If
            a positive integer is provided, the backbone is trained for one epoch, then frozen, and the
            predictors are trained alone for the specified number of epochs. This cycle is repeated
            throughout the training phase.
        """
        super().__init__()
        self.backbone = backbone
        self.projectors = projectors
        self.predictors = predictors
        self.loss_fn = loss_fn if loss_fn is not None else BatchWiseBarlowTwinLoss()
        self.adapter = adapter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.flatten = flatten

        self.predictor_training_epochs = predictor_training_epochs

    def _loss_func(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss between the output and the input data.

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
        loss = self.loss_fn(y_hat, y)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The predicted output and projected input.
        """
        z: torch.Tensor = self.backbone(x)
        if self.adapter:
            z = self.adapter(z)

        if self.flatten:
            z = z.view(z.size(0), -1)
            x = x.view(x.size(0), -1)

        y_pred = torch.stack([predictor(z) for predictor in self.predictors], 1)
        y_proj = torch.stack([projector(x) for projector in self.projectors], 1)

        return y_pred, y_proj

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        """
        Perform a single training/validation/test step.

        Parameters
        ----------
        batch : torch.Tensor
            The input batch of data.
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step (train, val, test).

        Returns
        -------
        torch.Tensor
            The loss value for the batch.
        """
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]
        x = batch
        y_pred, y_proj = self.forward(x)
        loss = 0
        for i in range(y_pred.shape[1]):
            loss += self._loss_func(y_pred[:, i, :], y_proj[:, i, :])
        loss /= len(y_pred)
        # loss = self._loss_func(y_pred, y_proj)
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="val")

    def on_train_epoch_start(self):
        if self.predictor_training_epochs and self.predictor_training_epochs > 0:
            freeze_backbone = (
                self.current_epoch % (self.predictor_training_epochs + 1) != 0
            )
            self.backbone.train(mode=(not freeze_backbone))

            for param in self.backbone.parameters():
                param.requires_grad = not freeze_backbone

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="test")

    def configure_optimizers(self):
        for param in self.projectors.parameters():
            param.requires_grad = False

        for proj in self.projectors:
            proj.eval()

        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
        )
