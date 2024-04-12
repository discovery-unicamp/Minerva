import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional
import torch
import lightning as L
from mlflow.exceptions import MlflowException
import fsspec
from io import BytesIO


class PickleBestModelAndLoad(L.Callback):
    def __init__(
        self,
        model_name: str,
        filename: str = "best_model.pt",
        model_tags: Optional[Dict[str, Any]] = None,
        model_description: Optional[str] = None,
    ):
        self.model_name = model_name
        self.filename = filename
        self.model_tags = model_tags
        self.model_description = model_description

    def on_train_end(
        self, trainer: L.Trainer, module: L.LightningModule
    ) -> None:
        # Check if it is rank 0
        if trainer.global_rank != 0:
            return

        # Does it have any
        if trainer.checkpoint_callback is not None:
            # Get the best model path
            best_model_path = getattr(
                trainer.checkpoint_callback, "best_model_path", None
            )

            if best_model_path is None:
                return

            # Load the model
            best_model_path = Path(best_model_path)
            # Load the best model checkpoint
            with fsspec.open(best_model_path, "rb") as f:
                model_bytes = f.read()
                model_bytes = BytesIO(model_bytes)
                module.load_state_dict(torch.load(model_bytes)["state_dict"])
            
            # Lets pickle the model
            with TemporaryDirectory(prefix="test", suffix="test") as tempdir:
                # Save the whle model in a temporary directory
                save_file = Path(tempdir) / self.filename
                torch.save(module, save_file)
                
                # Save the model as an MLFlow artifact
                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id, save_file, artifact_path=f"model"
                )

                # Locate the artifact path
                src = f"runs:/{trainer.logger.run_id}/model/{self.filename}"


                try:
                    trainer.logger.experiment.create_registered_model(
                        name=self.model_name,
                        tags={
                            "pickable": True
                        }
                    )
                except MlflowException:
                    pass
                
                trainer.logger.experiment.create_model_version(
                    name=self.model_name,
                    source=src,
                    run_id=trainer.logger.run_id,
                    tags=self.model_tags,
                    description=self.model_description,
                )
