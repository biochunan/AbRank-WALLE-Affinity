import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import rootutils
import torch
import wandb
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
from tqdm import tqdm

import waffle.constants as constants
from waffle import utils
from waffle.models.abrank_regression_gcn import RegressionGCNAbAgIntLM
from waffle.utils.wandb_helper import (log_config_as_artifact,
                                       log_default_root_dir, log_run_dir,
                                       upload_ckpts_to_wandb)

# ==================== Configuration ====================
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BASE = Path(__file__).parent

# set precision to trade off precision for performance
torch.set_float32_matmul_precision(precision="medium")


# ==================== Function ====================
def plot_pred_vs_true_affinity(
    y_pred: Tensor, y_true: Tensor, corr: Optional[float] = None
) -> plt.Figure:
    """
    Plot the predicted vs true affinity

    Args:
        y_pred (Tensor): predicted affinity
        y_true (Tensor): true affinity
        corr (Optional[float], optional): correlation between predicted and true affinity. Defaults to None.

    Returns:
        plt.Figure: plot of predicted vs true affinity
    """
    # move to cpu
    y_pred, y_true = y_pred.cpu(), y_true.cpu()
    # calculate the correlation between predicted and true affinity if not provided
    if corr is None:
        corr = PearsonCorrCoef()(y_pred, y_true).item()
    # plot
    fig, ax = plt.subplots()
    sns.regplot(
        x=y_true.numpy(), y=y_pred.numpy(), ci=None, scatter_kws={"alpha": 0.5}, ax=ax
    )
    ax.legend(
        [f"Pearson correlation: {corr:.3f}"],
        loc="best",
    )
    ax.set_xlabel("True affinity")
    ax.set_ylabel("Predicted affinity")
    ax.set_title("Predicted vs True affinity")
    ax.grid(True)
    return fig


def _num_training_steps(train_dataloader: PyGDataLoader, trainer: L.Trainer) -> int:
    """
    Returns total training steps inferred from datamodule and devices.

    Args:
        train_dataset: Training dataloader
        trainer: Lightning trainer

    Returns:
        Total number of training steps
    """
    if trainer.max_steps != -1:
        return trainer.max_steps

    dataset_size = (
        trainer.limit_train_batches
        if trainer.limit_train_batches not in {0, 1}
        else len(train_dataloader) * train_dataloader.batch_size
    )

    logger.info(f"Dataset size: {dataset_size}")

    num_devices = max(1, trainer.num_devices)
    effective_batch_size = (
        train_dataloader.batch_size * trainer.accumulate_grad_batches * num_devices
    )
    return (dataset_size // effective_batch_size) * trainer.max_epochs


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "/") -> Dict[str, any]:
    """
    Flatten a nested dictionary by concatenating keys with a separator.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        sep: Separator to use between keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_config_to_wandb(config: DictConfig, wandb_run: wandb.sdk.wandb_run.Run) -> None:
    """
    Log the config to wandb as columns.

    Args:
        config: Hydra config
        wandb_run: Wandb run object
    """
    if wandb_run is None:
        return

    # Convert DictConfig to regular dict and flatten it
    config_dict = dict(config)
    flat_config = flatten_dict(config_dict)

    # Log each config value as a column
    wandb_run.config.update(flat_config)


def train_model(cfg: DictConfig) -> None:
    """
    Trains a model from a config.

    If ``encoder`` is provided, it is used instead of the one specified in the
    config.

    - datamodule: instantiated from ``cfg.dataset.datamodule``
    - callbacks : instantiated from ``cfg.callbacks``
    - logger    : instantiated from ``cfg.logger``
    - trainer   : instantiated from ``cfg.trainer``
    - model     : instantiated from ``cfg.model``.
    - datamodule: is setup and a dummy forward pass is run to initialise
                  lazy layers for accurate parameter counts.
    - scheduler (Optional): If the config contains a scheduler, the number of
                            training steps is inferred from the datamodule and
                            devices and set in the scheduler.
    - Hyperparameters are logged to wandb if a logger is present.
    - The model is compiled if
        - ``cfg.compile`` is True.
        - ``cfg.task_name`` is ``"train"``.
        - ``cfg.test`` is ``True``.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    logger.info(f"Setting seed {cfg.seed} using Lightning.seed_everything")
    L.seed_everything(cfg.seed)

    # ----------------------------------------
    # Instantiate datamodule
    # ----------------------------------------
    logger.info(f"Instantiating datamodule: <{cfg.dataset.datamodule._target_}>...")
    dm: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )  # AbRankDataModule

    # setup and prepare data
    dm.prepare_data()
    dm.setup()

    # ----------------------------------------
    # Instantiate callbacks, loggers
    # ----------------------------------------
    logger.info("Instantiating callbacks...")
    callbacks, callback_names = utils.callbacks.instantiate_callbacks(
        cfg.get("callbacks")
    )

    logger.info("Instantiating loggers...")
    L_logger: List[Logger] = utils.loggers.instantiate_loggers(cfg.get("logger"))
    # if logger is wandb, initialise it via :func:experiment
    wandb_run = None
    run_name = uuid.uuid4().hex  # in case the wandb run is not initialized
    run_id = uuid.uuid4().hex  # in case the wandb run is not initialized
    for i in L_logger:
        if isinstance(i, WandbLogger):
            wandb_run = i.experiment  # this will initialize the wandb run
            # get wandb run id
            run_name = wandb_run.name  # e.g. "dulcet-sea-50"
            run_id = wandb_run.id  # e.g. "50"
            logger.info(f"Wandb run name: {run_name}")
            logger.info(f"Wandb run id: {run_id}")
            break

    # ----------------------------------------
    # Instantiate trainer
    # ----------------------------------------
    logger.info("Instantiating trainer...")
    logger.info(f"Trainer config: {cfg.trainer}")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=L_logger
    )

    # ----------------------------------------
    # Scheduler
    # ----------------------------------------
    if cfg.get("scheduler"):
        if (
            cfg.scheduler.scheduler._target_
            == "flash.core.optimizers.LinearWarmupCosineAnnealingLR"
            and cfg.scheduler.interval == "step"
        ):
            dm.setup()  # type: ignore
            num_steps = _num_training_steps(dm.train_dataloader(), trainer)  # => 10_000
            logger.info(
                f"Setting number of training steps in scheduler to: {num_steps}"
            )
            cfg.scheduler.scheduler.warmup_epochs = (
                trainer.val_check_interval
            )  # set to one epoch equivalent number of steps
            cfg.scheduler.scheduler.max_epochs = num_steps
            logger.info(cfg.scheduler)

    # ----------------------------------------
    # Model
    # ----------------------------------------
    logger.info("Instantiating model...")
    model: L.LightningModule = RegressionGCNAbAgIntLM(cfg=cfg)

    # ----------------------------------------
    # Model initialization
    # ----------------------------------------
    if cfg.model_init.method == "lazy_init":
        logger.info("Initializing lazy layers...")
        with torch.no_grad():
            dm.setup(stage="lazy_init")  # type: ignore
            batch = next(iter(dm.val_dataloader()))
            logger.info(f"Batch: {batch}")
            logger.info(f"Labels: {model.get_labels(batch)}")
            # forward pass
            out = model(batch)
            logger.info(f"Model output: {out}")
            del batch, out
    elif cfg.model_init.method == "from_pretrained_walle":
        # -----------------------------------------------------------
        # Load encoders' weights from
        # pre-trained classification model into regression model
        # -----------------------------------------------------------
        logger.info("Loading pretrained model weights...")
        assert (
            ckpt_path := Path(cfg.model_init.pretrained_model_ckpt)
        ).exists(), f"Model checkpoint {ckpt_path} does not exist"
        pret_model_dict = torch.load(ckpt_path, map_location=model.device)[
            "model_state_dict"
        ]
        model_dict = model.state_dict()
        for k, v in pret_model_dict.items():
            if "decoder" not in k:
                model_dict[k] = v
        model.load_state_dict(model_dict)
        logger.info(f"Loaded pretrained model weights from {ckpt_path}")

    # ----------------------------------------
    # Log config as artifact and columns
    # ----------------------------------------
    log_config_as_artifact(config=cfg, wandb_run=wandb_run)
    log_config_to_wandb(config=cfg, wandb_run=wandb_run)

    # ----------------------------------------
    # Training
    # ----------------------------------------
    if cfg.get("task_name") == "train":
        logger.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=dm,
            ckpt_path=cfg.get("ckpt_path"),  # resume from checkpoint
        )

    # ----------------------------------------
    # Testing
    # Evaluate the best model on the test set
    # ----------------------------------------
    # run test
    logger.info("Running test with the best model checkpoint...")
    # Get the test dataloader(s)
    test_loaders = dm.test_dataloader()
    # Store test dataloader(s) in the model for test_step to access
    if isinstance(test_loaders, dict):
        model.test_dataloader = test_loaders
    trainer.test(
        model=model,
        datamodule=dm,
        ckpt_path=trainer.checkpoint_callback.best_model_path,
    )

    # ----------------------------------------
    # Upload the best model to wandb
    # ----------------------------------------
    logger.info("Uploading checkpoints to wandb...")
    if cfg.get("upload_ckpts_to_wandb", True):
        # upload the best model to wandb if received sigterm or exit normally
        upload_ckpts_to_wandb(
            ckpt_callback=trainer.checkpoint_callback, wandb_run=wandb_run
        )
    logger.info("Uploading checkpoints to wandb... Done")

    # ----------------------------------------
    # Finishing
    # ----------------------------------------
    logger.info("Logging the run directory as an artifact...")
    try:
        log_default_root_dir(trainer.default_root_dir, wandb_run=wandb_run)
    except Exception as e:
        logger.error(f"Failed to log default_root_dir to wandb: {e}")
    try:
        log_run_dir(trainer.default_root_dir, wandb_run=wandb_run)
    except Exception as e:
        logger.error(f"Failed to log run_dir to wandb: {e}")
    logger.info("Logging the run directory as an artifact... Done")


def eval_model_ckpt_path(
    ckpt_path: str,
    eval_datamodule: L.LightningDataModule,
) -> Dict[str, float]:
    # load the model from the checkpoint
    model = RegressionGCNAbAgIntLM.load_from_checkpoint(ckpt_path)
    model.eval()

    # get the test dataloader(s)
    test_loaders = eval_datamodule.test_dataloader()

    results = {}

    # Handle both single dataloader and dictionary of dataloaders
    if isinstance(test_loaders, dict):
        # Multiple test datasets
        for test_name, loader in test_loaders.items():
            logger.info(f"Evaluating on test dataset: {test_name}")
            metrics = evaluate_on_dataloader(model, loader)
            for k, v in metrics.items():
                results[f"{test_name}/{k}"] = v
    else:
        # Single test dataset
        metrics = evaluate_on_dataloader(model, test_loaders)
        results.update(metrics)

    return results


def evaluate_on_dataloader(model, dataloader) -> Dict[str, float]:
    """Evaluate model on a single dataloader"""
    model.eval()

    all_y_pred = []
    all_y_true = []

    # evaluate the model on the whole dataloader
    with torch.no_grad():
        for batch in dataloader:
            batch1, batch2, ranking_labels = batch
            batch1 = batch1.to(model.device)
            batch2 = batch2.to(model.device)
            ranking_labels = ranking_labels.to(model.device)

            # forward pass
            y_pred1 = model(batch1)
            y_pred2 = model(batch2)
            label_pred = (y_pred1 > y_pred2).float() * 2 - 1

            # collect predictions and labels
            all_y_pred.append(label_pred)
            all_y_true.append(ranking_labels)

    # concatenate all batches
    y_pred = torch.cat(all_y_pred, dim=0).squeeze()
    y_true = torch.cat(all_y_true, dim=0).squeeze()

    # calculate metrics
    accuracy = (y_pred == y_true).float().mean().item()

    return {
        "accuracy": accuracy,
    }


# load hydra config from yaml files and command line arguments
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="train-abrank-regression",
)
def main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    # generate random seed if not specified i.e. cfg.seed was set null
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()
    logger.info(f"Random seed was not set, use random seed: {cfg.seed}")
    # generate random seed for datamodule if not specified i.e. cfg.dataset.datamodule.seed was set null
    if cfg.dataset.datamodule.seed is None:
        cfg.dataset.datamodule.seed = torch.randint(0, 2**32 - 1, (1,)).item()
    logger.info(
        f"Random seed was not set for datamodule, use random seed: {cfg.dataset.datamodule.seed}"
    )

    # train model
    train_model(cfg)


# ==================== Main ====================
if __name__ == "__main__":
    main()
