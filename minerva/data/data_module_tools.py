from typing import Optional, Tuple, Union

import lightning as L
import torch
from torch.utils.data import DataLoader


class SimpleDataset:
    def __init__(self, data: torch.Tensor, label: Optional[torch.Tensor] = None):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


class RandomDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_shape: Tuple[int, ...],
        label_shape: Union[int, Tuple[int, ...], None] = None,
        num_classes: Optional[int] = None,
        num_train_samples: int = 128,
        num_val_samples: int = 8,
        num_test_samples: int = 8,
        num_predict_samples: int = 8,
        batch_size: int = 8,
        data_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.num_classes = num_classes
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_predict_samples = num_predict_samples
        self.batch_size = batch_size
        self.data_dtype = data_dtype
        self.label_dtype = label_dtype

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None

        assert num_train_samples > 0, "num_train_samples must be greater than 0"

        if num_val_samples is not None:
            assert num_val_samples > 0, "num_val_samples must be greater than 0"
        else:
            delattr(self, "val_dataloader")

        if num_test_samples is not None:
            assert num_test_samples > 0, "num_test_samples must be greater than 0"
        else:
            delattr(self, "test_dataloader")

    def _generate_data(self, num_samples, data_shape, label_shape, num_classes):
        data = torch.rand((num_samples, *data_shape), dtype=self.data_dtype)
        label = None
        if label_shape is not None and num_classes is not None:
            label = torch.randint(0, num_classes, (num_samples, *label_shape))
        elif label_shape is not None:
            label = torch.rand((num_samples, *label_shape))
        elif num_classes is not None:
            label = torch.randint(0, num_classes, (num_samples,))

        label = label.to(dtype=self.label_dtype)

        return data, label

    def setup(self, stage):
        if stage == "fit":
            data, label = self._generate_data(
                self.num_train_samples,
                self.data_shape,
                self.label_shape,
                self.num_classes,
            )
            self.train_data = SimpleDataset(data, label)

            if self.num_val_samples is not None:
                data, label = self._generate_data(
                    self.num_val_samples,
                    self.data_shape,
                    self.label_shape,
                    self.num_classes,
                )
                self.val_data = SimpleDataset(data, label)

        elif stage == "test":
            if self.num_test_samples is not None:
                data, label = self._generate_data(
                    self.num_test_samples,
                    self.data_shape,
                    self.label_shape,
                    self.num_classes,
                )
                self.test_data = SimpleDataset(data, label)
        elif stage == "predict":
            if self.num_predict_samples is not None:
                data, label = self._generate_data(
                    self.num_predict_samples,
                    self.data_shape,
                    self.label_shape,
                    self.num_classes,
                )
                self.predict_data = SimpleDataset(data, label)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False)


def get_split_dataloader(data_module: L.LightningDataModule, stage: str) -> DataLoader:
    if stage == "train":
        data_module.setup("fit")
        return data_module.train_dataloader()
    elif stage == "validation":
        data_module.setup("fit")
        return data_module.val_dataloader()
    elif stage == "test":
        data_module.setup("test")
        return data_module.test_dataloader()
    elif stage == "predict":
        data_module.setup("predict")
        return data_module.predict_dataloader()
    else:
        raise ValueError(f"Invalid stage: {stage}")


def full_dataset_from_dataloader(dataloader: DataLoader):
    res = [dataloader.dataset[i] for i in range(len(dataloader.dataset))]
    # unpack the data and labels
    return list(zip(*res))


def get_full_data_split(
    data_module: L.LightningDataModule,
    stage: str,
):
    dataloader = get_split_dataloader(data_module, stage)
    return full_dataset_from_dataloader(dataloader)
