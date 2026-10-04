import csv
from unittest.mock import Mock

import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from minerva.callback.specific_checkpoint_callback import SpecificCheckpointCallback
from minerva.callback.synthetic_data_fid_callback import SyntheticDataFIDCallback


class SamplingModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0].transpose(1, 2)).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def sample(self, batch_size):
        return torch.randn(batch_size, 2, 8)


@pytest.fixture
def fid_training(tmp_path, monkeypatch):
    # Use fixed FID scores to test the callback without training TS2Vec.
    monkeypatch.setattr(
        "minerva.callback.synthetic_data_fid_callback.compute_ts_fid",
        Mock(side_effect=[4.0, 6.0, 1.0, 3.0]),
    )
    callback = SyntheticDataFIDCallback(
        every_n_train_steps=2, generation_batch_size=2, num_fid_runs=2
    )
    checkpoints = SpecificCheckpointCallback(specific_steps=[2, 4])
    dataset = TensorDataset(torch.rand(4, 2, 8), torch.zeros(4))
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=4,
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[checkpoints, callback],
    )
    trainer.fit(SamplingModel(), train_dataloaders=DataLoader(dataset, batch_size=2))
    return callback


def test_fid_callback_saves_generated_samples(fid_training):
    path = fid_training.output_dir / "synthetic_data_step=2.npy"

    samples = np.load(path)

    assert samples.shape == (4, 2, 8)


def test_fid_callback_evaluates_at_configured_steps(fid_training):
    path = fid_training.output_dir / fid_training.csv_filename
    with path.open() as stream:
        rows = list(csv.DictReader(stream))

    assert [row["step"] for row in rows] == ["2", "4"]


def test_fid_callback_saves_mean_score(fid_training):
    path = fid_training.output_dir / fid_training.csv_filename
    with path.open() as stream:
        first_result = next(csv.DictReader(stream))

    assert float(first_result["FID_score_mean"]) == pytest.approx(5.0)


def test_fid_callback_selects_checkpoint_with_lowest_score(fid_training):
    best = fid_training.checkpoint_dir / "best.ckpt"
    expected = fid_training.checkpoint_dir / "step=4.ckpt"

    assert best.read_bytes() == expected.read_bytes()


def test_fid_callback_restores_last_evaluated_step():
    callback = SyntheticDataFIDCallback()

    callback.load_state_dict({"last_step": 4})

    assert callback.state_dict() == {"last_step": 4}


def test_fid_callback_rejects_zero_interval():
    with pytest.raises(ValueError, match="every_n_train_steps"):
        SyntheticDataFIDCallback(every_n_train_steps=0)


def test_fid_callback_requires_multiple_fid_runs():
    with pytest.raises(ValueError, match="num_fid_runs"):
        SyntheticDataFIDCallback(num_fid_runs=1)
