"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from typing import Optional, Sequence
import torch
from torch.utils.data import Subset
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from src.data.dataset import SymDataset
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)



class DataModule(LightningDataModule):
    """`LightningDataModule` for training on 3D atomic datasets:


    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        debug: bool = False,
        use_all_data: bool = False,
        split_seed: int = 42,
        save_split_indices: bool = True,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.debug = debug
        self.use_all_data = use_all_data
        self.split_seed = split_seed
        self.save_split_indices = save_split_indices
        
        
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None




    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called AFTER
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        
        dataset = SymDataset(root = self.dataset.root, debug = self.debug)
        print(f'len dataset {len(dataset)}')

        dataset_name = self.dataset.root.split("/")[-1]

        
        if self.debug:
            num_samples = len(dataset)
            self.train_dataset = dataset[:num_samples]
            self.val_dataset = dataset[:num_samples]
            self.test_dataset = dataset[:num_samples]

            self.train_idx = torch.arange(num_samples)
            self.val_idx = torch.arange(num_samples)
            self.test_idx = torch.arange(num_samples)

        # NOTE Use all the data for training, no validation or testing
        elif self.use_all_data:
            self.train_dataset = dataset[:]
            self.val_dataset = None
            self.test_dataset = None

            self.train_idx = torch.arange(len(dataset))
            self.val_idx = None
            self.test_idx = None
        
        
        elif dataset_name == "mp20":
            # Original MP20 split
            # Note that we do not use test for anything
            self.train_idx = torch.arange(0, 27138)
            self.val_idx = torch.arange(27138, 27138 + 9046)
            self.test_idx = torch.arange(27138 + 9046, len(dataset))

            self.train_dataset = Subset(dataset, self.train_idx.tolist())
            self.val_dataset = Subset(dataset, self.val_idx.tolist())
            self.test_dataset = Subset(dataset, self.test_idx.tolist())
        
        
        elif dataset_name == "mpts52":
            # NOTE Around 30% overlap in train set between mp20 and mpts52
            self.train_idx = torch.arange(0, 27380)
            self.val_idx   = torch.arange(27380, 32380)
            self.test_idx  = torch.arange(32380, len(dataset))

            self.train_dataset = Subset(dataset, self.train_idx.tolist())
            self.val_dataset   = Subset(dataset, self.val_idx.tolist())
            self.test_dataset  = Subset(dataset, self.test_idx.tolist())
        
        elif dataset_name == "mp20_mpts52_merged":
            split_path = os.path.join(self.dataset.root, f"train_val_split.pt")

            split_dict = torch.load(split_path, weights_only=False)
            self.train_idx = split_dict["train_idx"]
            self.val_idx = split_dict["val_idx"]
            self.test_idx = None
            print(f"Loaded saved split from {split_path}")
        

            self.train_dataset = Subset(dataset, self.train_idx.tolist())
            self.val_dataset = Subset(dataset, self.val_idx.tolist())
            self.test_dataset = None

            print(f"Train: {len(self.train_dataset)}")
            print(f"Val:   {len(self.val_dataset)}")
            print(f"Test:  0")
        
        
        else:
            raise Exception("Picked wrong dataset")


    
        if stage is None or stage in ["fit", "validate"]:
            self.train_dataset = ConcatDataset([self.train_dataset])
            
            log.info(f"Train dataset: {len(self.train_dataset)} samples")

            if self.val_dataset is not None:
                log.info(f"Validation dataset: {len(self.val_dataset)} samples")
            else:
                log.info("MP20 validation dataset: None (use_all_data=True)")

            if self.test_dataset is not None:
                log.info(f"Test dataset: {len(self.test_dataset)} samples")
            else:
                log.info("Test dataset: None")


        
        if stage is None or stage in ["test", "predict"]:
            log.info(f"Test dataset: {len(self.test_dataset)} samples")

        

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """

        if self.val_dataset is None:
            return []

        return [
            DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=False,
                shuffle=False,
            )
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

        if self.test_dataset is None:
            return []

        return [
            DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=False,
                shuffle=False,
            ),
        ]
    

    