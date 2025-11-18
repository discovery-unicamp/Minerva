import lightning as L
import torch
import torch.nn.functional as F
from minerva.losses.batchwise_barlowtwins_loss import BatchWiseBarlowTwinLoss
from typing import Optional, Callable
import numpy as np
import math


# -------------------------------------------------------------------------------------------------
# This function was extracted from the LFR implementation in
# https://github.com/layer6ai-labs/lfr/blob/main/utils/dpp.py
def dpp(kernel_matrix, max_length, epsilon=1e-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


# -------------------------------------------------------------------------------------------------


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
    A PyTorch Lightning model for pretraining with the technique 'Learning From Random Projectors'.
    When using 'predictor_training_epochs', please consider updating your number of training epochs as well.
    Otherwise, the LFR backbone will be trained for less epochs:
    - If the total training epochs in your Trainer is 100, and 'predictor_training_epochs' is 1, then the
      backbone will be trained on the epochs 0, 2, 4, 6, ... and 98, resulting in the backbone being effectively
      trained for 50 epochs instead of the specified 100.
    - If the total training epochs in your Trainer is 100, and 'predictor_training_epochs' is 2, then the
      backbone will be trained on the epochs 0, 3, 6, 9, ... and 99, resulting in the backbone being effectively
      trained for 34 epochs instead of the specified 100.
    In conclusion, consider updating your total number of training epochs to:
      Total number of training epochs = (intended backbone training epochs) * (predictor_training_epochs + 1)

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
        loss_fn: Optional[torch.nn.Module] = None,
        num_targets: Optional[int] = None,
        adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        flatten: bool = False,
        predictor_training_epochs: Optional[int] = None,
        max_backbone_training_steps: Optional[int] = None,
        selection_batch_size: int = 128,
    ):
        """
        Initialize the LFR_Model, freezing the projectors. Remember to update your number of training epochs when
        using 'predictor_training_epochs':
          Total number of training epochs = (intended backbone training epochs) * (predictor_training_epochs + 1)

        Parameters
        ----------
        backbone: torch.nn.Module
            The backbone neural network for feature extraction.
        projectors: torch.nn.ModuleList
            A list of projector networks.
        predictors: torch.nn.ModuleList
            A list of predictor networks.
        num_targets: Optional[int]
            The number of projectors and predictors to select from the lists provided, using the Fast
            Determinantal Point Process (DPP) algorithm. All projectors and predictors are used if the
            value received is None, a negative integer, or an integer greater than the length of the lists.
        loss_fn: Optional[torch.nn.Module]
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
        max_backbone_training_steps: Optional[int]
            The number of steps the backbone will be trained, by default None. If None, zero, or
            negative, no limit is applied. The steps where the backbone is frozen are ignored.
        selection_batch_size: int
            By default 128. When selecting projectors and predictors, this variable decides how many
            random samples from the dataset are used in the Fast Determinantal Point Process (DPP)
            algorithm.
        """
        super().__init__()
        self.backbone = backbone
        self.projectors = projectors
        self.predictors = predictors
        self.num_targets = num_targets
        self.loss_fn = loss_fn if loss_fn is not None else BatchWiseBarlowTwinLoss()
        self.adapter = adapter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.flatten = flatten
        self.selection_batch_size = selection_batch_size

        self.predictor_training_epochs = predictor_training_epochs
        if self.predictor_training_epochs is None or self.predictor_training_epochs < 0:
            self.predictor_training_epochs = 0

        self.max_backbone_training_steps = max_backbone_training_steps
        if (
            self.max_backbone_training_steps is None
            or self.max_backbone_training_steps < 0
        ):
            self.max_backbone_training_steps = 0

        # Helper variables
        self.backbone_training_steps_counter = 1  # Because steps start at 1
        self.freeze_backbone = False

        for param in self.projectors.parameters():
            param.requires_grad = False

        for proj in self.projectors:
            proj.eval()

        # Parameter restrictions
        if self.num_targets is None and len(self.projectors) != len(self.predictors):
            raise ValueError(
                "When num_targets is None, the number of projectors and predictors must be equal."
            )
        if self.num_targets is not None and (
            self.num_targets <= 0
            or self.num_targets > min(len(self.projectors), len(self.predictors))
        ):
            raise ValueError(
                "num_targets must be None or a positive integer less than or equal to the number of projectors and predictors."
            )

    def setup(self, stage):
        """
        Setup function. If necessary, it picks projectors and predictors based on 'num_targets'
        using the first 128 elements of the training dataset, as used in
        https://github.com/layer6ai-labs/lfr/blob/main/scripts/har/run_har_diet.sh.
        """
        if stage != "fit":
            return
        if self.num_targets is None:
            return
        # Get the training dataset
        training_dataset = self.trainer.datamodule.train_dataloader().dataset
        # Pick random values from the dataset
        random_idxs = np.random.choice(
            range(len(training_dataset)),
            size=self.selection_batch_size,
            replace=self.selection_batch_size > len(training_dataset),
        )
        sample_batch = []
        for random_idx in random_idxs:
            random_sample = training_dataset[random_idx]
            if isinstance(random_sample, (tuple, list)):
                random_sample = random_sample[0]
            sample_batch.append(random_sample)
        sample_batch = np.array(sample_batch)
        sample_batch = torch.from_numpy(sample_batch)
        self._select_targets(sample_batch)

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

    def _loss_from_targets(self, y_pred: torch.Tensor, y_proj: torch.Tensor):
        """
        Computes the average loss between each pair of predictor and projector outputs.
        This function is isolated from `_single_step` to make it easier to test
        independently.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predictions tensors.
        y_proj : torch.Tensor
            The projections tensors.
        """
        loss = torch.tensor(0)
        for i in range(y_pred.shape[1]):
            loss = loss + self.loss_fn(y_pred[:, i, :], y_proj[:, i, :])
        loss /= y_pred.shape[1]
        return loss

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        """
        Perform a single training/validation/test step, computing and logging the loss.

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
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        x = batch
        y_pred, y_proj = self.forward(x)
        loss = self._loss_from_targets(y_pred=y_pred, y_proj=y_proj)
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def _select_targets(self, sample_data):
        """
        Select projectors and predictors based on 'num_targets' using the Fast Determinantal Point
        Process (DPP) algorithm and some sample data. Code adapted from
        https://github.com/layer6ai-labs/lfr/blob/main/ssl_models/lfr.py
        """
        with torch.no_grad():
            sims = []
            for projector in self.projectors:
                # (bs, dim)
                rep = projector(sample_data)
                if rep.shape[0] > 1000:
                    rep = rep[
                        np.random.RandomState(seed=42).permutation(
                            np.arange(rep.shape[0])
                        )[:1000]
                    ]
                rep_normalized = F.normalize(rep, dim=1)
                # (bs, bs) cosine similarity
                sim = rep_normalized @ rep_normalized.T
                sims.append(sim.view(-1))
            # N, bs^2
            sims = torch.stack(sims)
            sims_normalized = F.normalize(sims, dim=1)
            # N,N
            sims_targets = sims_normalized @ sims_normalized.T
            result = dpp(sims_targets.cpu().numpy(), self.num_targets)
        # Select the projectors and predictors based on the result from DPP
        self.projectors = torch.nn.ModuleList([self.projectors[idx] for idx in result])
        if len(self.predictors) != self.num_targets:
            self.predictors = torch.nn.ModuleList(
                [self.predictors[idx] for idx in range(self.num_targets)]
            )

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Perform a training step using the '_single_step' method.

        Parameters
        ----------
        batch : torch.Tensor
            The input batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss value for the batch.
        """
        return self._single_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Perform a validation step using the '_single_step' method.

        Parameters
        ----------
        batch : torch.Tensor
            The input batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss value for the batch.
        """
        return self._single_step(batch, batch_idx, step_name="val")

    def on_train_epoch_start(self):
        """
        Executed at the start of each training epoch. If the predictor training epochs is valid, this
        function evaluates the current epoch number and freeze or unfreeze the backbone based on it.
        If the predictor training epochs is None, zero, or negative, the backbone is always trained.
        In the first epoch, the backbone is trained. In the subsequent 'predictor_training_epochs'
        epochs, it is frozen.
        """
        if self.predictor_training_epochs > 0:
            self.freeze_backbone = (
                self.current_epoch % (self.predictor_training_epochs + 1) != 0
            )
            self.backbone.train(mode=(not self.freeze_backbone))

            for param in self.backbone.parameters():
                param.requires_grad = not self.freeze_backbone

    def on_train_batch_start(self, batch, batch_idx):
        """
        If a training steps limit is set, it checks the training step counter at the start of every
        training batch. If the counter reached the limit, it returns -1, stopping the training.
        """
        if (
            self.max_backbone_training_steps > 0
            and self.backbone_training_steps_counter > self.max_backbone_training_steps
        ):
            return -1

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Updates the backbone training steps counter only if the backbone is not frozen.
        """
        if not self.freeze_backbone:
            self.backbone_training_steps_counter += 1

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        This method sets up the optimizer for the model's parameters, excluding the projectors.
        """

        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
        )
