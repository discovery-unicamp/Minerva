from minerva.models.ssl.lfr import LearnFromRandomnessModel
from minerva.models.nets.lfr_har_architectures import (
    HARSCnnEncoder,
    LFR_HAR_Projector,
    LFR_HAR_Predictor,
)
import torch
import pytest
import lightning as L
from tests.pipelines.test_lightning_pipeline import MyDataModule
from minerva.callback.specific_checkpoint_callback import SpecificCheckpointCallback
import os
from pathlib import Path


def test_lfr_har():
    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=256, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [
                LFR_HAR_Projector(encoding_size=256, input_channel=6, middle_dim=544)
                for _ in range(8)
            ]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None


def test_lfr_har_adapter():
    example_adapter = torch.nn.Linear(300, 256)
    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=300, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [
                LFR_HAR_Projector(encoding_size=256, input_channel=6, middle_dim=544)
                for _ in range(8)
            ]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
        adapter=example_adapter,
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None


def test_lfr_erroneous_adapter_input():
    example_adapter = torch.nn.Linear(301, 256)
    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=300, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [
                LFR_HAR_Projector(encoding_size=256, input_channel=6, middle_dim=544)
                for _ in range(8)
            ]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
        adapter=example_adapter,
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    with pytest.raises(RuntimeError):
        y = model(x)


def test_lfr_erroneous_adapter_output():
    example_adapter = torch.nn.Linear(300, 255)
    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=300, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [
                LFR_HAR_Projector(encoding_size=256, input_channel=6, middle_dim=544)
                for _ in range(8)
            ]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
        adapter=example_adapter,
    )
    assert model is not None

    x = torch.rand(32, 6, 60)
    with pytest.raises(RuntimeError):
        y = model(x)


def test_num_targets():
    example_adapter = torch.nn.Linear(300, 256)
    with pytest.raises(
        ValueError,
        match="When num_targets is None, the number of projectors and predictors must be equal.",
    ):
        model = LearnFromRandomnessModel(
            backbone=HARSCnnEncoder(
                dim=300, input_channel=6, inner_conv_output_dim=128 * 10
            ),
            projectors=torch.nn.ModuleList(
                [
                    LFR_HAR_Projector(
                        encoding_size=256, input_channel=6, middle_dim=544
                    )
                    for _ in range(10)
                ]
            ),
            predictors=torch.nn.ModuleList(
                [
                    LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                    for _ in range(8)
                ]
            ),
            num_targets=None,
            loss_fn=None,
            learning_rate=1e-3,
            flatten=False,
            predictor_training_epochs=7,
            adapter=example_adapter,
        )

    with pytest.raises(
        ValueError,
        match="num_targets must be None or a positive integer less than or equal to the number of projectors and predictors.",
    ):
        model = LearnFromRandomnessModel(
            backbone=HARSCnnEncoder(
                dim=300, input_channel=6, inner_conv_output_dim=128 * 10
            ),
            projectors=torch.nn.ModuleList(
                [
                    LFR_HAR_Projector(
                        encoding_size=256, input_channel=6, middle_dim=544
                    )
                    for _ in range(10)
                ]
            ),
            predictors=torch.nn.ModuleList(
                [
                    LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                    for _ in range(8)
                ]
            ),
            num_targets=-1,
            loss_fn=None,
            learning_rate=1e-3,
            flatten=False,
            predictor_training_epochs=7,
            adapter=example_adapter,
        )

    with pytest.raises(
        ValueError,
        match="num_targets must be None or a positive integer less than or equal to the number of projectors and predictors.",
    ):
        model = LearnFromRandomnessModel(
            backbone=HARSCnnEncoder(
                dim=300, input_channel=6, inner_conv_output_dim=128 * 10
            ),
            projectors=torch.nn.ModuleList(
                [
                    LFR_HAR_Projector(
                        encoding_size=256, input_channel=6, middle_dim=544
                    )
                    for _ in range(10)
                ]
            ),
            predictors=torch.nn.ModuleList(
                [
                    LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                    for _ in range(8)
                ]
            ),
            num_targets=9,
            loss_fn=None,
            learning_rate=1e-3,
            flatten=False,
            predictor_training_epochs=7,
            adapter=example_adapter,
        )

    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=300, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [
                LFR_HAR_Projector(encoding_size=256, input_channel=6, middle_dim=544)
                for _ in range(10)
            ]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(8)
            ]
        ),
        num_targets=8,
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
        adapter=example_adapter,
    )

    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None

    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=300, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [
                LFR_HAR_Projector(encoding_size=256, input_channel=6, middle_dim=544)
                for _ in range(20)
            ]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(encoding_size=256, middle_dim=128, num_layers=3)
                for _ in range(5)
            ]
        ),
        num_targets=4,
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=7,
        adapter=example_adapter,
    )

    # Simple trainer
    trainer = L.Trainer(
        max_epochs=10,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        callbacks=[],
    )
    # Variables
    num_samples, channels, timestamps = 100, 6, 60
    # Simple dataset
    random_x = torch.rand((num_samples, channels, timestamps))
    dataset = torch.utils.data.TensorDataset(random_x)
    datamodule = MyDataModule(dataset=dataset, batch_size=64)
    # Simple training
    trainer.fit(model=model, datamodule=datamodule)
    assert model is not None

    x = torch.rand(32, 6, 60)
    y = model(x)
    assert y is not None


def test_max_backbone_training_steps():
    # Variables
    num_samples, channels, timestamps, model_encoding_size = 100, 6, 60, 10
    # Simple dataset
    random_x = torch.rand((num_samples, channels, timestamps))
    dataset = torch.utils.data.TensorDataset(random_x)
    datamodule = MyDataModule(dataset=dataset, batch_size=64)

    class CustomCallback(L.Callback):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.counter += 1

    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=model_encoding_size, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [
                LFR_HAR_Projector(
                    encoding_size=model_encoding_size, input_channel=6, middle_dim=544
                )
                for _ in range(2)
            ]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(
                    encoding_size=model_encoding_size, middle_dim=128, num_layers=3
                )
                for _ in range(2)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=0,
        adapter=None,
        max_backbone_training_steps=3,
    )
    # Counter callback
    counter_callback = CustomCallback()
    # Simple trainer
    trainer = L.Trainer(
        max_epochs=10,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        callbacks=[counter_callback],
    )
    # Simple training
    trainer.fit(model=model, datamodule=datamodule)

    assert counter_callback.counter == model.max_backbone_training_steps

    model = LearnFromRandomnessModel(
        backbone=HARSCnnEncoder(
            dim=model_encoding_size, input_channel=6, inner_conv_output_dim=128 * 10
        ),
        projectors=torch.nn.ModuleList(
            [
                LFR_HAR_Projector(
                    encoding_size=model_encoding_size, input_channel=6, middle_dim=544
                )
                for _ in range(2)
            ]
        ),
        predictors=torch.nn.ModuleList(
            [
                LFR_HAR_Predictor(
                    encoding_size=model_encoding_size, middle_dim=128, num_layers=3
                )
                for _ in range(2)
            ]
        ),
        loss_fn=None,
        learning_rate=1e-3,
        flatten=False,
        predictor_training_epochs=1,
        adapter=None,
        max_backbone_training_steps=3,
    )
    # Counter callback
    counter_callback = CustomCallback()
    checkpoint_callback = SpecificCheckpointCallback(
        specific_steps=[0, 1, 2, 3], step_var_name="backbone_training_steps_counter"
    )
    # Simple trainer
    trainer = L.Trainer(
        max_epochs=10,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        callbacks=[counter_callback, checkpoint_callback],
    )
    # Simple training
    trainer.fit(model=model, datamodule=datamodule)

    assert counter_callback.counter == model.max_backbone_training_steps + 2
    assert not os.path.exists(Path(trainer.log_dir) / "checkpoints" / "step=0.ckpt")
    assert os.path.exists(Path(trainer.log_dir) / "checkpoints" / "step=1.ckpt")
    assert os.path.exists(Path(trainer.log_dir) / "checkpoints" / "step=2.ckpt")
    assert os.path.exists(Path(trainer.log_dir) / "checkpoints" / "step=3.ckpt")
