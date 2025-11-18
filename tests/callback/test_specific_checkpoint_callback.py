import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
import os
from pathlib import Path
from minerva.callback.specific_checkpoint_callback import SpecificCheckpointCallback


class MyDataModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )


class MyModel(L.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return torch.rand(size=(1,), requires_grad=True)

    def configure_optimizers(self):
        return super().configure_optimizers()


def test_basic_checkpoints_har_like():
    # Callback definition
    callback = SpecificCheckpointCallback(
        specific_epochs=[-1, 0, 1], specific_steps=[1, 2]
    )
    x = torch.rand(size=(100, 6, 60))
    y = torch.randint(low=0, high=6, size=(len(x),))

    dataset = TensorDataset(x, y)
    backbone = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=6, out_channels=8, kernel_size=3),
        torch.nn.Flatten(),
        torch.nn.Linear(20, 6),
    )
    model = MyModel(backbone=backbone)
    dm = MyDataModule(dataset, batch_size=64)
    trainer = L.Trainer(
        max_epochs=3,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        callbacks=[callback],
    )

    trainer.fit(model=model, datamodule=dm)

    assert os.path.exists(Path(trainer.log_dir) / "checkpoints" / "epoch=-1.ckpt")
    assert os.path.exists(Path(trainer.log_dir) / "checkpoints" / "epoch=0.ckpt")
    assert os.path.exists(Path(trainer.log_dir) / "checkpoints" / "epoch=1.ckpt")
    assert os.path.exists(Path(trainer.log_dir) / "checkpoints" / "step=1.ckpt")
    assert os.path.exists(Path(trainer.log_dir) / "checkpoints" / "step=2.ckpt")
