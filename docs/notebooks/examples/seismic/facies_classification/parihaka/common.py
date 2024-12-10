import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import lightning as L
from matplotlib import pyplot as plt
from torchmetrics import Accuracy, JaccardIndex

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline

from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


from minerva.data.data_modules.parihaka import (
    ParihakaDataModule,
    default_train_transforms,
    default_test_transforms,
)
from minerva.utils.typing import PathLike
from typing import Tuple, Optional
from minerva.transforms.transform import Indexer, Unsqueeze, Squeeze


def get_data_module(
    root_data_dir: PathLike,
    root_annotation_dir: PathLike,
    img_size: Tuple[int, int] = (1006, 590),
    batch_size: int = 1,
    num_workers: Optional[int] = None,
    seed: int = 42,
    single_channel: bool = False,
) -> L.LightningDataModule:
    train_transforms = default_train_transforms(img_size=img_size, seed=seed)
    if single_channel:
        train_transforms[0] += Indexer(0)
        train_transforms[0] += Unsqueeze(0)
        train_transforms[1] += Indexer(0)
        train_transforms[1] += Unsqueeze(0)

    test_transforms = default_test_transforms(img_size=img_size, seed=seed)
    if single_channel:
        test_transforms[0] += Indexer(0)
        test_transforms[0] += Unsqueeze(0)
        test_transforms[1] += Indexer(0)
        test_transforms[1] += Unsqueeze(0)

    test_transforms[1] += Squeeze(0)

    return ParihakaDataModule(
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        train_transforms=train_transforms,
        valid_transforms=train_transforms,
        test_transforms=test_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_trainer(
    model_name: str,
    dataset_name: str,
    log_dir: PathLike = "logs",
    num_epochs: int = 100,
    accelerator: str = "gpu",
    devices: int = 1,
    is_debug: bool = False,
) -> L.Trainer:
    logger = CSVLogger(log_dir, name=model_name, version=dataset_name)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, save_last=True)
    max_epochs = num_epochs if not is_debug else 3
    limit_train_batches = 3 if is_debug else None
    limit_val_batches = 3 if is_debug else None
    limit_test_batches = 3 if is_debug else None
    limit_predict_batches = 3 if is_debug else None
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        limit_predict_batches=limit_predict_batches,
    )

    return trainer


def get_trainer_pipeline(
    model: L.LightningModule,
    model_name: str,
    dataset_name: str,
    log_dir: PathLike = "logs",
    num_epochs: int = 100,
    accelerator: str = "gpu",
    devices: int = 1,
    is_debug: bool = False,
    seed: int = 42,
) -> SimpleLightningPipeline:
    trainer = get_trainer(
        model_name=model_name,
        dataset_name=dataset_name,
        log_dir=log_dir,
        num_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        is_debug=is_debug,
    )
    log_dir = Path(log_dir) / model_name / dataset_name
    return SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
        seed=seed,
    )


def get_evaluation_pipeline(
    model: L.LightningModule,
    model_name: str,
    dataset_name: str,
    log_dir: PathLike = "logs",
    accelerator: str = "gpu",
    devices: int = 1,
    is_debug: bool = False,
    seed: int = 42,
) -> SimpleLightningPipeline:
    trainer = get_trainer(
        model_name=model_name,
        dataset_name=dataset_name,
        log_dir=log_dir,
        num_epochs=0,
        accelerator=accelerator,
        devices=devices,
        is_debug=is_debug,
    )

    log_dir = Path(log_dir) / model_name / dataset_name / "evaluation"
    return SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=False,
        seed=seed,
        classification_metrics={
            "mIoU": JaccardIndex(
                num_classes=6, average="macro", task="multiclass"
            ),
            "acc": Accuracy(num_classes=6, task="multiclass"),
        },
        save_predictions=True,
    )


def plot_image(img, title=None, cmap: str = "gray"):
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.show()


def plot_images(
    images,
    plot_title=None,
    subplot_titles=None,
    cmaps=None,
    filename=None,
    x_label=None,
    y_label=None,
    height=5,
    width=5,
    show=False,
):
    num_images = len(images)

    # Create a figure with subplots (1 row, num_images columns), adjusting size based on height and width parameters
    fig, axs = plt.subplots(1, num_images, figsize=(width * num_images, height))

    # Set overall plot title if provided
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16)

    # Ensure subplot_titles and cmaps are lists with correct lengths
    if subplot_titles is None:
        subplot_titles = [None] * num_images
    if cmaps is None:
        cmaps = ["gray"] * num_images

    # Plot each image in its respective subplot
    for i, (img, ax, title, cmap) in enumerate(
        zip(images, axs, subplot_titles, cmaps)
    ):
        im = ax.imshow(img, cmap=cmap)

        # Set title for each subplot if provided
        if title is not None:
            ax.set_title(title)

        # Add a colorbar for each subplot
        fig.colorbar(im, ax=ax)

        # Set x and y labels if provided
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

    # Adjust layout to fit titles, labels, and colorbars
    plt.tight_layout()

    # Save the figure if filename is provided
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Figure saved as '{filename}'")

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()
