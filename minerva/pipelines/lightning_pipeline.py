from typing import Literal
import lightning as L
from minerva.pipelines.base import Pipeline


class SimpleLightningPipeline(Pipeline):
    def __init__(
        self,
        model: L.LightningModule,
        trainer: L.Trainer,
        cwd: str = None,
        save_run_status: bool = False,
    ):
        super().__init__(
            cwd=cwd,
            ignore=["model", "trainer"],
            cache_result=True,
            save_run_status=save_run_status,
        )
        self._model = model
        self._trainer = trainer
        self._data = None

    # Public read-only properties
    @property
    def model(self):
        return self._model

    @property
    def trainer(self):
        return self._trainer

    @property
    def data(self):
        return self._data

    # Private methods
    def _fit(self, data: L.LightningDataModule, **trainer_kwargs):
        return self._trainer.fit(self._model, data, **trainer_kwargs)

    def _test(self, data: L.LightningDataModule, **trainer_kwargs):
        return self._trainer.test(self._model, data, **trainer_kwargs)

    def _predict(self, data: L.LightningDataModule, **trainer_kwargs):
        return self._trainer.predict(self._model, data, **trainer_kwargs)

    def _evaluate(self, data: L.LightningDataModule, **trainer_kwargs):
        return self._predict(self._model, data, **trainer_kwargs)

    # Default run method (entry point)
    def _run(
        self,
        data: L.LightningDataModule,
        task: Literal["fit", "test", "predict", "evaluate"] = "fit",
        **trainer_kwargs,
    ):
        self._data = data

        if task == "fit":
            return self._fit(data, **trainer_kwargs)
        elif task == "test":
            return self._test(data, **trainer_kwargs)
        elif task == "predict":
            return self._predict(data, **trainer_kwargs)
        elif task == "evaluate":
            return self._evaluate(data, **trainer_kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")


def main():
    from jsonargparse import CLI

    CLI(SimpleLightningPipeline, as_positional=False)


if __name__ == "__main__":
    main()
