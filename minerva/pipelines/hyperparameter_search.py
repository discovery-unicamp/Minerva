from typing import Dict, Optional

import lightning as L
from torchmetrics import Metric

from minerva.pipelines.base import Pipeline
from minerva.utils.typing import PathLike


class HyperParameterSearch(Pipeline):

    def __init__(
        self,
        model: L.LightningModule,
        trainer: L.Trainer,
        log_dir: PathLike,
        save_run_status: bool,
        classification_metrics: Dict[str, Metric],
        regression_metrics: Dict[str, Metric],
        apply_metrics_per_sample: bool,
    ):
        self.model = model
        self.trainer = trainer
        self.log_dir = log_dir
        self.save_run_status = save_run_status
        self.classification_metrics = classification_metrics
        self.regression_metrics = regression_metrics
        self.apply_metrics_per_sample = apply_metrics_per_sample
