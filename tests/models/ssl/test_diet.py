import torch
from tests.pipelines.test_lightning_pipeline import MyDataModule
from minerva.models.ssl.diet import DIET
import pytest
import lightning as L


@pytest.mark.parametrize("num_samples", [100, 200])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_diet_basic(num_samples, batch_size):
    # Variables
    features, model_encoding_size = 50, 10
    # Simple dataset
    random_x = torch.rand((num_samples, features))
    data_index = torch.arange(0, len(random_x))
    dataset = torch.utils.data.TensorDataset(random_x, data_index)
    datamodule = MyDataModule(dataset=dataset, batch_size=batch_size)
    # Simple DIET model
    simple_backbone = torch.nn.Linear(features, model_encoding_size)
    linear_head = torch.nn.Linear(model_encoding_size, len(random_x))
    model = DIET(
        backbone=simple_backbone,
        linear_head=linear_head,
        num_data=None,
        flatten=True,
        adapter=None,
        loss=None,
        learning_rate=3e-4,
        weight_decay=3e-4,
        wca_scheduler_total_epochs=None,
    )
    # Simple trainer
    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )
    # Simple training
    trainer.fit(model=model, datamodule=datamodule)


@pytest.mark.parametrize("num_samples", [50, 100])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_diet_without_linear_head(num_samples, batch_size):
    # Variables
    features, model_encoding_size = 50, 10
    # Simple dataset
    random_x = torch.rand((num_samples, features))
    data_index = torch.arange(0, len(random_x))
    dataset = torch.utils.data.TensorDataset(random_x, data_index)
    datamodule = MyDataModule(dataset=dataset, batch_size=batch_size)
    # Simple DIET model
    simple_backbone = torch.nn.Linear(features, model_encoding_size)
    model = DIET(
        backbone=simple_backbone,
        linear_head=None,
        num_data=None,
        flatten=True,
        adapter=None,
        loss=None,
        learning_rate=3e-4,
        weight_decay=3e-4,
        wca_scheduler_total_epochs=None,
    )
    # Simple trainer
    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )
    # Simple training
    trainer.fit(model=model, datamodule=datamodule)

    assert model.linear_head is not None
    assert model.linear_head.in_features == model_encoding_size
    assert model.linear_head.out_features == len(dataset)


@pytest.mark.parametrize("num_samples", [50, 100])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_diet_with_wrong_linear(num_samples, batch_size):
    # Variables
    features, model_encoding_size = 50, 10
    # Simple dataset
    random_x = torch.rand((num_samples, features))
    data_index = torch.arange(0, len(random_x))
    dataset = torch.utils.data.TensorDataset(random_x, data_index)
    datamodule = MyDataModule(dataset=dataset, batch_size=batch_size)
    # Simple DIET model
    simple_backbone = torch.nn.Linear(features, model_encoding_size)

    # WARNING CASE
    # The linear head output exceeds dataset length
    linear_head = torch.nn.Linear(model_encoding_size, len(random_x) + 1)
    model = DIET(
        backbone=simple_backbone,
        linear_head=linear_head,
        num_data=None,
        flatten=True,
        adapter=None,
        loss=None,
        learning_rate=3e-4,
        weight_decay=3e-4,
        wca_scheduler_total_epochs=None,
    )
    # Simple trainer
    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )
    # Simple training
    with pytest.raises(
        AssertionError,
        match=f"Number of samples\\({num_samples}\\) and output of linear head\\({linear_head.out_features}\\) do not match.",
    ):
        trainer.fit(model=model, datamodule=datamodule)

    # The linear head output exceeds dataset length and num_data is provided
    linear_head = torch.nn.Linear(model_encoding_size, len(random_x) + 1)
    model = DIET(
        backbone=simple_backbone,
        linear_head=linear_head,
        num_data=len(random_x),
        flatten=True,
        adapter=None,
        loss=None,
        learning_rate=3e-4,
        weight_decay=3e-4,
        wca_scheduler_total_epochs=None,
    )
    # Simple trainer
    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )
    # Simple training
    with pytest.raises(
        AssertionError,
        match=f"Number of samples\\({num_samples}\\) and output of linear head\\({linear_head.out_features}\\) do not match.",
    ):
        trainer.fit(model=model, datamodule=datamodule)

    # ERROR CASE
    # The linear head output is less than dataset length
    linear_head = torch.nn.Linear(model_encoding_size, len(random_x) - 1)
    model = DIET(
        backbone=simple_backbone,
        linear_head=linear_head,
        num_data=None,
        flatten=True,
        adapter=None,
        loss=None,
        learning_rate=3e-4,
        weight_decay=3e-4,
        wca_scheduler_total_epochs=None,
    )
    # Simple trainer
    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )
    # Simple training
    with pytest.raises(
        AssertionError,
        match=f"Number of samples\\({num_samples}\\) and output of linear head\\({linear_head.out_features}\\) do not match.",
    ):
        trainer.fit(model=model, datamodule=datamodule)

    # The linear head output is less than dataset length and num_data is provided
    linear_head = torch.nn.Linear(model_encoding_size, len(random_x) - 1)
    model = DIET(
        backbone=simple_backbone,
        linear_head=linear_head,
        num_data=len(random_x),
        flatten=True,
        adapter=None,
        loss=None,
        learning_rate=3e-4,
        weight_decay=3e-4,
        wca_scheduler_total_epochs=None,
    )
    # Simple trainer
    trainer = L.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
    )
    # Simple training
    with pytest.raises(
        AssertionError,
        match=f"Number of samples\\({num_samples}\\) and output of linear head\\({linear_head.out_features}\\) do not match.",
    ):
        trainer.fit(model=model, datamodule=datamodule)
