from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from typing import List


class EmbeddingLoggerCallback(Callback):

    def __init__(
        self,
        data_X: Tensor,
        logger: CSVLogger,
        data_Y: Tensor = None,
        feature_preffix: str = "EMB-",
        backbone_names_list: List[str] = ["backbone", "encoder"],
    ) -> None:
        """
        Callback to extract and log embeddings from some data using the model's backbone.

        Parameters
        ----------
        data_X : torch.Tensor
            Tensor with the input data.
        logger : CSVLogger, optional
            The logger to use.
        data_Y : torch.Tensor, optional
            Tensor with the target data, by default None.
        feature_preffix : str, optional
            The preffix to use for the feature names, by default 'EMB-'.
        backbone_names_list : List[str], optional
            List with the names of the backbones in the model, by default ['backbone', 'encoder'].
        """
        super().__init__()
        self.data_X = data_X
        self.data_Y = data_Y
        self.logger = logger
        self.feature_preffix = feature_preffix
        self.backbone_names_list = backbone_names_list

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        filtered_names_list = [
            name for name in self.backbone_names_list if hasattr(pl_module, name)
        ]
        if len(filtered_names_list) == 0:
            raise ValueError("No backbone found in the model")
        self.backbone = getattr(pl_module, filtered_names_list[0])

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Obtaining the embeddings
        self.backbone.eval()
        embeddings = self.backbone(self.data_X).detach().cpu().numpy()
        self.backbone.train()
        # Logging the embeddings
        for row_index, row in enumerate(embeddings):
            data_dict = {
                f"{self.feature_preffix}{str(index).zfill(3)}": value
                for index, value in enumerate(row)
            }
            data_dict["epoch"] = trainer.current_epoch
            if self.data_Y is not None:
                data_dict["y"] = self.data_Y[row_index].item()
            self.logger.log_metrics(data_dict)
