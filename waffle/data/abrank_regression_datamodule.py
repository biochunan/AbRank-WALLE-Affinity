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

from waffle.data.components.abrank_dataset_regression import AbRankDataset
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
        exclude_keys: List[str] = ["metadata", "edge_index_bg", "y_b", "y_g"],
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
        data_registry_idx: Tensor,
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
        self.data_registry_idx = data_registry_idx
        """e.g.
        self.data_registry_idx = tensor([
            21,
            24,
            33,
            35,
            ...
        ])

        These are the indices of items in the dataset data registry to form
        the dataset, not necessary to start from 0
        """

    def __len__(self) -> int:
        return len(self.data_registry_idx)

    def get_pair(self, idx: int) -> PairData:
        """
        Get a pair of antibody-antigen pairs
        idx is the numerical index of items in `self.data_registry_idx`
        NOT the base dataset index
        """
        idx = self.data_registry_idx[idx]
        graph_pair: PairData = self.base_dataset.get(idx)
        return graph_pair

    def __getitem__(self, idx: int) -> PairData:
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
        exclude_keys: List[str] = ["metadata", "edge_index_bg", "y_b", "y_g"],
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
        self.train_split_path = train_split_path
        self.test_split_path_dict = test_split_path_dict

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
        # extract the dbID i.e. 1st element of each element in self.dataset.data_registry
        # as a dictionary for quick lookup and forming train and test splits
        self.dbID2idx = {
            dbID: i for i, (dbID, _, _) in enumerate(self.dataset.data_registry)
        }
        # --- splits ---
        self.split_dict = self.load_split_dict(
            split_paths={
                "train": self.train_split_path,
                "test": self.test_split_path_dict,
            }
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
                data_registry_idx=self.split_dict["train"],
            )
        elif stage in ["val", "lazy_init"]:
            self.val_dataset = AbRankSampler(
                base_dataset=self.dataset,
                seed=self.seed,
                data_registry_idx=self.split_dict["val"],
            )
        elif stage in ["test", "predict"]:
            self.test_dataset = {}
            for split_name, data_registry_idx in self.split_dict["test"].items():
                self.test_dataset[split_name] = AbRankSampler(
                    base_dataset=self.dataset,
                    seed=self.seed,
                    data_registry_idx=data_registry_idx,
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
            exclude_keys=self.exclude_keys,  # default ["metadata", "edge_index_bg", "y_b", "y_g"]
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

    def _load_train_split(self, csv_path: str) -> List[int]:
        """
        Load the split from the given path, extract (dbID1, dbID2, deltaLogKD)
        Example input:
        dbID,abName,agName,srcDB,logAff,affOp,fileName,fileExists,setType
        0,Ab-AIntibody-002,SARS-CoV-2,AINTIBODY,-1.4776,=,Ab-AIntibody-002---SARS-CoV-2-pairdata.pt,True,train
        1,Ab-AIntibody-003,SARS-CoV-2,AINTIBODY,-1.4711,=,Ab-AIntibody-003---SARS-CoV-2-pairdata.pt,True,train
        """
        df = pd.read_csv(filepath_or_buffer=csv_path)
        dbID_list = df["dbID"].to_list()
        # find the corresponding index in self.dbID2idx
        data_list_idx = [self.dbID2idx[dbID] for dbID in dbID_list]
        return data_list_idx

    def _load_test_split(self, csv_path: str) -> List[int]:
        """
        Load the test split from the given path, extract dbID
        Example input:
        dbID1,dbID2,deltaLogKD,op,fileName1,fileName2,combName
        59612,61,-2.599,=,AbOVA-029---OVA-pairdata.pt,Ab-AIntibody-072---SARS-CoV-2-pairdata.pt,AbOVA-029---OVA:::Ab-AIntibody-072---SARS-CoV-2
        61,59643,1.502,=,Ab-AIntibody-072---SARS-CoV-2-pairdata.pt,AbOVA-063---OVA-pairdata.pt,Ab-AIntibody-072---SARS-CoV-2:::AbOVA-063---OVA
        58082,68,-1.054,=,abHIV-PCIN71M1a---HIV-ZM214_15-pairdata.pt,Ab-AIntibody-079---SARS-CoV-2-pairdata.pt,abHIV-PCIN71M1a---HIV-ZM214_15:::Ab-AIntibody-079---SARS-CoV-2
        """
        df = pd.read_csv(csv_path)
        dbID1_list = df["dbID1"].to_list()
        dbID2_list = df["dbID2"].to_list()
        # find the corresponding index in self.dbID2idx
        data_list_idx1 = [self.dbID2idx[dbID] for dbID in dbID1_list]
        data_list_idx2 = [self.dbID2idx[dbID] for dbID in dbID2_list]
        # combine and remove duplicates
        data_list_idx = sorted(set(data_list_idx1 + data_list_idx2))
        return data_list_idx

    def _select_val_set(
        self, train_idx: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Select 10% of the data from training set to serve as val

        Args:
            train_idx (List[int]): A list of indices, where each index is the
                index of the training set, this is the raw training set of length `N`

        Returns:
            train_idx (List[int]): A list of indices, where each index is the
                index of the training set, of length `n_train` (90% of `N`)
            val_idx (List[int]): A list of indices, where each index is the
                index of the validation set, of length `n_val` (10% of `N`)
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # total number of training samples
        n = len(train_idx)

        # Calculate number of validation samples (10% of total)
        n_val = int(n * 0.1)

        # use numpy to select random indices
        val_indices = np.random.permutation(n)[:n_val]
        train_indices = np.setdiff1d(np.arange(n), val_indices)

        # Convert to list and select validation samples
        val_set = [train_idx[i] for i in val_indices]

        # remove val_set from train_idx
        train_set = [train_idx[i] for i in train_indices]

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
        split_dict["train"] = self._load_train_split(split_paths["train"])
        # load test splits
        split_dict["test"] = {
            split_name: self._load_test_split(path)
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
    DATA_PATH = os.getenv("DATA_PATH")
    dm = AbRankDataModule(
        root=DATA_PATH,
        train_split_path=osp.join(
            DATA_PATH,
            "AbRank",
            "processed",
            "splits-regression",
            "Split_AF3",
            "balanced-train-regression.csv",
        ),
        test_split_path_dict={
            "generalization": osp.join(
                DATA_PATH,
                "AbRank",
                "processed",
                "splits-regression",
                "Split_AF3",
                "test-generalization-swapped.csv",
            ),
            "perturbation": osp.join(
                DATA_PATH,
                "AbRank",
                "processed",
                "splits-regression",
                "Split_AF3",
                "test-perturbation-swapped.csv",
            ),
        },
        seed=42,
        num_workers=4,
        batch_size=32,
    )

    dm.prepare_data()
    dm.setup(stage="train")
    dm.setup(stage="val")
    dm.setup(stage="test")

    print(len(dm.val_dataset))  # 6929
    print(len(dm.train_dataset))  # 52365
    print(len(dm.test_dataset))  # 2 loaders
    print(len(dm.test_dataset["generalization"]))  # 152
    print(len(dm.test_dataset["perturbation"]))  # 185

    train_dataloader = dm.train_dataloader()
    print(len(train_dataloader))  # 1949 batches
    val_dataloader = dm.val_dataloader()
    print(len(val_dataloader))  # 217 batches
    test_dataloader = dm.test_dataloader()
    print(len(test_dataloader))  # 2 loaders
    print(len(test_dataloader["generalization"]))  # 5 batches
    print(len(test_dataloader["perturbation"]))  # 6 batches

    # extract a batch
    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))
    test_batch = next(iter(test_dataloader["generalization"]))

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
