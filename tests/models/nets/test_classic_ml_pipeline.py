from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader
import lightning as L
from minerva.models.nets.classic_ml_pipeline import ClassicMLModel, SklearnPipeline
from minerva.data.data_module_tools import SimpleDataset

def test_sklearn_pipeline():
    X, y = make_blobs(n_samples=64, centers=2, random_state=42)
    train_dataset = SimpleDataset(X, y)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    pipeline = ClassicMLModel(
        backbone=torch.nn.Identity(),
        head=SklearnPipeline(
            [
                ["min-max", {"class_path": "sklearn.preprocessing.MinMaxScaler", "init_args": {}}],
                ["log-reg", {"class_path": "sklearn.linear_model.LogisticRegression", "init_args": {"random_state": 42, "max_iter": 5}}]
            ]
        )
    )
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(pipeline, train_dataloader)
    