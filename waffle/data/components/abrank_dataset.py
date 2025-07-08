"""
AbRank dataset

This dataset is used to load the AbRank dataset.
"""

import os
import os.path as osp
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import gdown
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
from dotenv import load_dotenv
from loguru import logger
from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from waffle.data.components.pair_data import PairData

# ==================== Config ====================
logger.configure(
    handlers=[{"sink": RichHandler(rich_tracebacks=True), "format": "{message}"}]
)


# ==================== Configuration ====================
BASE = Path(__file__).parent.resolve()
# NOTE: make sure you have .projct-root under your project root path


# ==================== Function ====================
class AbRankDataset(PygDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        Args:
            root (str): Root directory where the dataset should be saved
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
            split_paths (Dict[str, Path]): The paths to the split files.
                keys: train, test
                values: paths to the corresponding split files
                e.g.
                {
                    "train": Path("/path/to/balanced-train-split.csv"),
                    "test": Path("/path/to/balanced-test-split.csv"),
                }
        """
        self.name: str = "AbRank"
        super().__init__(str(root), transform, pre_transform, pre_filter)
        self.pairdata_dir = osp.join(
            self.processed_dir, "ca_10", "pairdata"
        )  # directory containing pairdata files e.g. "Ab-AIntibody-002---SARS-CoV-2-pairdata.pt"
        self.data_registry_path = osp.join(
            self.processed_dir, "registry", "AbRank-all.csv"
        )  # maps dbID to fileName
        self.data_registry = self.load_data_registry(
            self.data_registry_path
        )  # dbID -> fileName

    def __repr__(self) -> str:
        return f"{self.name}({len(self)} AbAg Pairs)"

    def __len__(self) -> int:
        return len(self.data_registry)

    @property
    def raw_dir(self) -> str:
        """return the raw directory"""
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """return the processed directory"""
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        """return the raw file names"""
        # NOTE: N/A, because the graph pairs are pre-downloaded
        return [
            "registry.tar.gz",
            "splits.tar.gz",
            "md5sum.txt",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        """return the processed file names"""
        return [
            # registry file, maps dbID to file name e.g. 0 => "Ab-AIntibody-002---SARS-CoV-2-pairdata.pt"
            # the pt file is located in `self.processed_dir/ca_10/pairdata/`
            osp.join("processed", "registry", "AbRank-all.csv"),
            # Split_AF3 files
            osp.join("processed", "splits", "Split_AF3", "balanced-train-split.csv"),
            osp.join("processed", "splits", "Split_AF3", "balanced-train-swapped.csv"),
            osp.join("processed", "splits", "Split_AF3", "hard-ab-train.csv"),
            osp.join("processed", "splits", "Split_AF3", "hard-ab-train-swapped.csv"),
            osp.join("processed", "splits", "Split_AF3", "hard-ag-train.csv"),
            osp.join("processed", "splits", "Split_AF3", "hard-ag-train-swapped.csv"),
            osp.join("processed", "splits", "Split_AF3", "test-generalization.csv"),
            osp.join(
                "processed", "splits", "Split_AF3", "test-generalization-swapped.csv"
            ),
            osp.join("processed", "splits", "Split_AF3", "test-perturbation.csv"),
        ]

    def process(self):
        # NOTE: N/A, because the graph pairs are pre-processed
        # with their file path stored in `data_registry`
        pass

    def download(self):
        FILEIDS = {
            "md5sum.txt": "1Zv-YQhZIT00pkYryV3mVT3SC05o2yt8x",
            "registry.tar.gz": "1ZVkFUqyebwd_j6o9PPRl0PVHqjAkitZp",
            "splits.tar.gz": "1dL4D5ScjZhpL1KmnEIPw2FbFjADxRAyy",
        }
        # First download md5sum.txt
        logger.info("Downloading md5sum.txt...")
        file_id = FILEIDS["md5sum.txt"]
        md5sum_path = osp.join(self.raw_dir, "md5sum.txt")
        gdown.download(id=file_id, output=md5sum_path, quiet=False)

        # Download files
        files_to_download = [
            # (file_id, output_name)
            (FILEIDS["registry.tar.gz"], "registry.tar.gz"),
            (FILEIDS["splits.tar.gz"], "splits.tar.gz"),
        ]

        for file_id, output_name in files_to_download:
            output_path = osp.join(self.raw_dir, output_name)
            logger.info(f"Downloading {output_name}...")
            gdown.download(id=file_id, output=output_path, quiet=False)

        # Validate downloaded files using md5sum.txt
        try:
            from waffle.utils.file_validation import validate_downloaded_files

            logger.info("Validating downloaded files...")
            validated, missing, corrupted = validate_downloaded_files(
                download_dir=self.raw_dir, md5sum_path=md5sum_path
            )

            if missing or corrupted:
                logger.error(
                    f"Validation failed: {len(missing)} missing, {len(corrupted)} corrupted files"
                )
                logger.error("Please re-run the download or manually fix the issues")
                # Don't raise an exception here to allow the process to continue even with corrupted files
                # Users will see the error message and can decide what to do
            else:
                logger.info(f"All {len(validated)} files validated successfully")
        except ImportError:
            logger.warning("File validation module not found. Skipping validation.")
        except Exception as e:
            logger.error(f"Error during file validation: {e}")

        # Extract files
        for _, output_name in files_to_download:
            tar_path = osp.join(self.raw_dir, output_name)
            logger.info(f"Extracting {output_name} to {self.processed_dir}...")
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(self.processed_dir)

    def get(self, idx: int) -> PairData:
        """
        Get the graph pair

        Args:
            idx (int): The index (i.e. `dbID` in the data registry) of the graph pair

        Returns:
            PairData: The graph pair
        """
        # get the file path from the data registry
        # e.g. idx:=0 -> "Ab-AIntibody-002---SARS-CoV-2-pairdata.pt"
        filename = self.data_registry[idx]
        # load the graph pair
        return torch.load(osp.join(self.pairdata_dir, filename))

    def load_data_registry(self, data_registry_path: str) -> Dict[int, str]:
        """
        Load the data registry from the given path to `AbRank_all.csv`
        Example lines:
        ```csv
        dbID,abName,agName,abagID,srcDB,fileName,filePath
        0,Ab-AIntibody-002,SARS-CoV-2,1_100000,AINTIBODY,Ab-AIntibody-002---SARS-CoV-2-pairdata.pt,/path/to/Ab-AIntibody-002---SARS-CoV-2-pairdata.pt
        1,Ab-AIntibody-003,SARS-CoV-2,1_100000,AINTIBODY,Ab-AIntibody-003---SARS-CoV-2-pairdata.pt,/path/to/Ab-AIntibody-003---SARS-CoV-2-pairdata.pt
        2,Ab-AIntibody-004,SARS-CoV-2,1_100000,AINTIBODY,Ab-AIntibody-004---SARS-CoV-2-pairdata.pt,/path/to/Ab-AIntibody-004---SARS-CoV-2-pairdata.pt
        ...
        ```
        """
        try:
            assert osp.isfile(data_registry_path)
        except AssertionError:
            raise FileNotFoundError(
                f"Data registry file not found at {data_registry_path}"
            )
        df = pd.read_csv(data_registry_path)
        # set dbID type to int
        df["dbID"] = df["dbID"].astype(int)
        # order by dbID
        df = df.sort_values(by="dbID", ascending=True)
        # return a dictionary key: dbID, value: fileName
        return df.set_index("dbID")["fileName"].to_dict()


if __name__ == "__main__":
    load_dotenv("/workspaces/AbRank-WALLE-Affinity/.env")
    PROJECT_ROOT = os.getenv("ROOT_DIR")
    root = osp.join(PROJECT_ROOT, "data", "local", "api")  # type:ignore
    ds = AbRankDataset(root=root)
