import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from loguru import logger
from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.loader import DataLoader as PyGDataLoader

from waffle.data.components.abrank_dataset import AbRankDataset
from waffle.data.components.pair_data import PairData

# ==================== Config ====================
logger.configure(
    handlers=[{"sink": RichHandler(rich_tracebacks=True), "format": "{message}"}]
)


# ==================== Function ====================
class AbRankDataCollator:
    """
    Collator for ranking data
    """

    def __init__(
        self,
        follow_batch: List[str] = ["x_b", "x_g"],
        exclude_keys: List[str] = ["metadata", "edge_index_bg"],
    ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(
        self, batch: List[Tuple[PairData, PairData, Tensor]]
    ) -> Tuple[PyGBatch, PyGBatch, Tensor]:
        """
        Args:
            batch: List of tuples (g1, g2, label) where
                g1, g2 are PairData objects
                label is a tensor indicating which graph has higher affinity

        Returns:
            g1_batch: PyG Batch object containing all g1 graphs
            g2_batch: PyG Batch object containing all g2 graphs
            labels: Tensor of shape (batch_size,) containing the labels
        """
        # Unzip the batch into separate lists
        g1s, g2s, labels = zip(*batch)

        # Create PyG batch objects for g1 and g2 graphs
        g1_batch = PyGBatch.from_data_list(
            g1s, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys
        )
        g2_batch = PyGBatch.from_data_list(
            g2s, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys
        )

        # Stack the labels into a single tensor
        labels_batch = torch.stack(labels)

        return g1_batch, g2_batch, labels_batch


class AbRankBase(TorchDataset):
    """
    This class is used to create a base dataset of antibody-antigen pairs
    """

    def __init__(self, base_dataset: AbRankDataset, seed: int = 42):
        """
        Args:
            base_dataset (AbRankDataset): The base dataset to sample from
            seed (int): The seed to use for random number generation
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.seed = seed

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[PairData, PairData]:
        """
        Get a pair of antibody-antigen pairs
        """
        pass

    @abstractmethod
    def get_pair(self, idx1: int, idx2: int) -> Tuple[PairData, PairData, Tensor]:
        """
        Get a combination of two antibody-antigen pairs with a label
        Binary label:
        - 1 if the affinity value of g1 is greater than g2,
        - 0 otherwise
        """
        pass


class AbRankSampler(AbRankBase):
    """
    This class is used to sample a random pair of antibody-antigen pairs
    """

    def __init__(
        self,
        base_dataset: AbRankDataset,
        combinations: Tensor,
        seed: int = 42,
    ):
        """
        Args:
            base_dataset (AbRankDataset): The base dataset to sample from
            seed (int): The seed to use for random number generation
            combinations (Tensor): It's a tensor of shape [N x 3], where N is the
                number of combinations of two antibody-antigen pairs
                and the 3 columns are (dbID1, dbID2, deltaLogKD)
        """
        super().__init__(base_dataset, seed)
        self.combinations = combinations
        """e.g.
        self.combinations = tensor([
            [21,1147,1.269],
            [24,1003,1.218],
            [33,1106,-1.003],
            [35,1151,1.162],
            ...
        ])

        columns:
        1. `dbID1` dbID of the 1st antibody-antigen pair, same index in data registry
        2. `dbID2` dbID of the 2nd antibody-antigen pair, same index in data registry
        3. `deltaLogKD` is the deltaLogKD of the combination (KD of g1 - KD of g2)
        """

    def __len__(self) -> int:
        return len(self.combinations)

    def get_pair(self, idx: int) -> Tuple[PairData, PairData, Tensor]:
        """
        Get a pair of antibody-antigen pairs
        """
        dbID1, dbID2, deltaLogKD = self.combinations[idx]
        g1 = self.base_dataset.get(dbID1)
        g2 = self.base_dataset.get(dbID2)
        # NOTE: deltaLogKD is the deltaLogKD of the combination (KD of g1 - KD of g2)
        # NOTE: this is for ranking task, not regression task
        label = (
            torch.tensor(1).type(torch.int64)
            if deltaLogKD > 0
            # NOTE: required by MarginRankingLoss that negative label is -1 NOT 0
            else torch.tensor(-1).type(torch.int64)
        )
        return g1, g2, label

    def __getitem__(self, idx: int) -> Tuple[PairData, PairData, Tensor]:
        """
        Get a pair of antibody-antigen pairs
        """
        return self.get_pair(idx)


# ----------------------------------------
# turn into lightning data module
# ----------------------------------------
class AbRankDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        # dataset config
        root: str,
        data_registry_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        # split config
        train_split_path: str,
        test_split_path_dict: Dict[str, Dict[str, str]],
        # kwargs for dataset config
        seed: int = 42,
        name: str = "AbRank",
        # lightning config
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # batching config
        batch_size: int = 32,
        shuffle: bool = True,
        # PyG DataLoader config
        follow_batch: List[str] = ["x_b", "x_g"],
        exclude_keys: List[str] = ["metadata", "edge_index_bg", "y_b", "y_g", "y"],
    ):
        """
        Args:
            # ----- dataset config -----
            root (str): Root directory where AbRank dataset is stored, AbRank
            dataset will be stored in
            root/
            └── AbRank/
                ├── raw/
                │   ├── ...
                │   └── ...
                └── processed/
                    ├── registry/
                    │   └── AbRank-all.csv
                    └── splits/
                        ├── Split_Crystallized/
                        │   ├── ...
                        │   └── ...
                        └── Split_AF3/
                            ├── balanced-train.csv
                            ├── balanced-test-split.csv
                            ├── hard-ab-train.csv
                            ├── hard-ag-train.csv
                            ├── test-generalization.csv
                            └── test-perturbation.csv
            data_registry_path (str): The path to the data registry file
            transform (Callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                Post-transform after loading each item.
                (default: :obj:`None`)
            pre_transform (Callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. Transform before saving.
                (default: :obj:`None`)
            pre_filter (Callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. Filter before saving.
                (default: :obj:`None`)
            seed (int): The seed to use for random number generation
            # ----- split config -----
            train_split_path (str): The path to the training split file, e.g.
                train_split_path = "/path/to/balanced-train-split.csv"
            test_split_path_dict (Dict[str, str]): The path to the test split file, e.g.
                test_split_path_dict = {
                    "generalization": "/path/to/test-generalization.csv",
                    "perturbation": "/path/to/test-perturbation.csv",
                }
            # ----- lightning config -----
            num_workers (int): The number of workers to use for loading data
            batch_size (int): The batch size to use for training,
                NOTE: val and test batch size can be specified as "all" to load
                      entire subset as a single batch, use the `batch_size` arg
                      in `self.val_dataloader()` and `self.test_dataloader()`
            shuffle (bool): Whether to shuffle the data
            follow_batch (List[str]): The keys to follow when batching
            exclude_keys (List[str]): The keys to exclude when batching
        """
        super().__init__()
        # --- AbRank dataset config ---
        self.name = name
        self.root = root
        self.data_registry_path = data_registry_path
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.seed = seed
        # --- lightning config ---
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        # but if num_workers == 0, pin_memory and persistent_workers are ignored
        if self.num_workers == 0:
            self.pin_memory = False
            self.persistent_workers = False
        # --- PyG DataLoader config ---
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        # --- collator config ---
        self.collator = AbRankDataCollator(
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
        )

        # --- splits ---
        self.split_dict = self.load_split_dict(
            split_paths={
                "train": train_split_path,
                "test": test_split_path_dict,
            }
        )

    def prepare_data(self):
        """
        Initialize the base dataset that indexes single antibody-antigen pairs
        E.g.
        self.dataset.get(i) -> PairData
        """
        # This is the base dataset that indexes single pair data
        self.dataset = AbRankDataset(
            root=self.root,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
        )

    def setup(self, stage: Optional[str] = None):
        """
        Load and split data

        This takes the base dataset and turns it into a ranking dataset
        The ranking dataset returns a tuple of (g1, g2, label) for data index i
        """
        if stage is None:
            self.setup(stage="train")
            self.setup(stage="val")
            self.setup(stage="test")
        elif stage in ["fit", "train"]:
            self.train_dataset = AbRankSampler(
                base_dataset=self.dataset,
                seed=self.seed,
                combinations=self.split_dict["train"],
            )
        elif stage in ["val", "lazy_init"]:
            self.val_dataset = AbRankSampler(
                base_dataset=self.dataset,
                seed=self.seed,
                combinations=self.split_dict["val"],
            )
        elif stage in ["test", "predict"]:
            self.test_dataset = {}
            for split_name, combinations in self.split_dict["test"].items():
                self.test_dataset[split_name] = AbRankSampler(
                    base_dataset=self.dataset,
                    seed=self.seed,
                    combinations=combinations,
                )
        else:
            raise ValueError(
                f"Invalid stage: {stage}. Choices: [None, 'fit', 'train', 'val', 'test', 'predict']"
            )

    def train_dataloader(self):
        # collator is a callable object
        # to turn [(g1_i, g2_i, label_i) for i in range(batch_size)]
        # into a tuple of (g1_batch, g2_batch, labels_batch)

        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,  # default 32
            shuffle=self.shuffle,
            num_workers=self.num_workers,  # default 4
            follow_batch=self.follow_batch,  # default ["x_b", "x_g"]
            exclude_keys=self.exclude_keys,  # default ["metadata"]
            collate_fn=self.collator,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """
        2025 March 27: We keep the validation dataloader a sampling dataloader

        2025 April 2:
        - if use_batcher is True, the validation dataloader will generate all possible pairs that delta_y_min < abs(delta_y)
            - if apply_delta_y is True, the pairs will be filtered by delta_y_min < abs(delta_y) < delta_y
        - if use_batcher is False, the validation dataloader will sample pairs from the dataset
        """
        if self.val_dataset is None:
            self.setup(stage="val")
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
            collate_fn=self.collator,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """
        Same configuration as the validation dataloader
        """
        if self.test_dataset is None:
            self.setup(stage="test")
        # for split_name, dataset in self.test_dataset.items():
        return {
            split_name: PyGDataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
                collate_fn=self.collator,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
            for split_name, dataset in self.test_dataset.items()
        }  # split_name => `generalization`, `perturbation`

    @staticmethod
    def _load_split(csv_path: str) -> List[Tuple[int, int, float]]:
        """
        Load the split from the given path, extract (dbID1, dbID2, deltaLogKD)
        """
        df = pd.read_csv(csv_path)
        # TODO: do we need `op` column?
        return list(
            zip(
                df["dbID1"],
                df["dbID2"],
                df["deltaLogKD"],  # NOTE: serves as label
            )
        )

    def _select_val_set(
        self, combinations: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """
        Select 10% of the data from training set to serve as val

        Args:
            combinations (List[Tuple[int, int, float]]): A list of tuples,
                where each tuple is (dbID1, dbID2, deltaLogKD)
                this is the raw training set of length `N`

        Returns:
            val_idx (List[Tuple[int, int, float]]): A list of tuples, where
                each tuple is (dbID1, dbID2, deltaLogKD)
                of length `n_val` (10% of `N`)
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # total number of combinations
        n = len(combinations)

        # Calculate number of validation samples (10% of total)
        n_val = int(n * 0.1)

        # use numpy
        val_indices = np.random.permutation(n)[:n_val]
        train_indices = np.setdiff1d(np.arange(n), val_indices)

        # Convert to list and select validation samples
        val_set = [combinations[i] for i in val_indices]

        # remove val_set from combinations
        train_set = [combinations[i] for i in train_indices]

        return train_set, val_set

    def load_split_dict(self, split_paths: Dict[str, str]) -> Dict[str, Tensor]:
        """
        # NOTE: in the dataset split, there is no validation set
        Load the split dictionary from the given paths
        Example input:
        {
            "train": "/path/to/balanced-train-split.csv",
            "test": {
                "generalization": "/path/to/balanced-test-split.csv",
                "perturbation": "/path/to/balanced-test-split.csv",
            },
        }
        Args:
            split_paths (Dict[str, Dict[str, str]]): The paths to the split files.
                keys: train, test
                values: paths to the corresponding split files
        """
        assert "train" in split_paths.keys(), "Training set is required"
        split_dict: Dict[str, Tensor] = {}
        # load train split
        split_dict["train"] = self._load_split(split_paths["train"])
        # load test splits
        split_dict["test"] = {
            split_name: self._load_split(path)
            for split_name, path in split_paths["test"].items()
        }

        # split 10% from training set to serve as val
        split_dict["train"], split_dict["val"] = self._select_val_set(
            split_dict["train"]
        )
        return split_dict


# ==================== Main ====================
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("/workspaces/WALLE-Affinity/.env")
    ROOT_DIR = os.getenv("ROOT_DIR")
    root = osp.join(ROOT_DIR, "data", "local", "api")
    dm = AbRankDataModule(
        root=root,
        train_split_path=osp.join(root, "AbRank", "processed", "splits", "Split_AF3", "balanced-train.csv"),
        test_split_path_dict={
            "generalization": osp.join(root, "AbRank", "processed", "splits", "Split_AF3", "test-generalization.csv"),
            "perturbation": osp.join(root, "AbRank", "processed", "splits", "Split_AF3", "test-perturbation.csv"),
        },
        seed=42,
        num_workers=4,
        batch_size=256,
    )

    dm.prepare_data()
    dm.setup(stage="train")
    dm.setup(stage="val")
    dm.setup(stage="test")

    print(len(dm.val_dataset))  # 16846
    print(len(dm.train_dataset))  # 151623
    print(len(dm.test_dataset))  # 2 loaders
    print(len(dm.test_dataset["generalization"]))  # 753
    print(len(dm.test_dataset["perturbation"]))  # 345

    train_dataloader = dm.train_dataloader()
    print(len(train_dataloader))  # 4739 batches
    val_dataloader = dm.val_dataloader()
    print(len(val_dataloader))  # 527 batches
    test_dataloader = dm.test_dataloader()
    print(len(test_dataloader))  # 2 loaders
    print(len(test_dataloader["generalization"]))  # 24 batches
    print(len(test_dataloader["perturbation"]))  # 11 batches

    # record the time to make a batch
    import time
    for i in range(10):
        start_time = time.time()
        batch = next(iter(train_dataloader))
        end_time = time.time()
        print(f"{end_time - start_time:.4f} seconds")
        # print(batch[0])
        # print(batch[1])
        # print(batch[2])
