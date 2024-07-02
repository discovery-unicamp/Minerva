from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from minerva.data.datasets.har_rodrigues_dataset import HARDatasetCPC

# Defining the data loader for the implementation
class HARDataModuleCPC(LightningDataModule):
    def __init__(self, root_dir, data_file = "RealWorld_raw", input_size = 6, window = 60, overlap = 30, batch_size = 64):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = HARDatasetCPC(root_dir, data_file, input_size, window, overlap, phase='train')
        self.val_dataset = HARDatasetCPC(root_dir, data_file, input_size, window, overlap, phase='val')
        self.test_dataset = HARDatasetCPC(root_dir, data_file, input_size, window, overlap, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)