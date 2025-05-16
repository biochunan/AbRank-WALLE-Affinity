import atexit
import signal
import tarfile
import tempfile
from functools import wraps
from pathlib import Path
from types import FrameType
from typing import Any, Dict, Optional, Union

import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from wandb.wandb_run import Run


def signal_handler_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def handle_exit_signal(signum: int, frame: Optional[FrameType]) -> None:
            logger.info(f"Received signal {signum}, executing decorated function...")
            func(*args, **kwargs)
            exit(code=0)  # Exit cleanly after handling the signal

        # Register signal handlers
        signal.signal(
            signal.SIGINT, handle_exit_signal
        )  # Handle keyboard interrupt (Ctrl+C)
        signal.signal(signal.SIGTERM, handle_exit_signal)  # Handle termination signal

        # Register with atexit to handle normal program termination
        atexit.register(func, *args, **kwargs)

        # Call the original function
        return func(*args, **kwargs)

    return wrapper


@signal_handler_decorator
def upload_ckpts_to_wandb(
    ckpt_callback: ModelCheckpoint, wandb_run: Optional[Run] = None
) -> None:
    """
    Upload the best ckpt and the last ckpt to wandb.
    """
    # model paths
    best_model_path = ckpt_callback.best_model_path
    last_model_path = ckpt_callback.last_model_path
    # create a new artifact
    artifact = wandb.Artifact(name="checkpoints", type="model")
    name_best_ckpt = f"BEST-{Path(best_model_path).name}"
    artifact.add_file(best_model_path, name=name_best_ckpt)
    name_last_ckpt = f"LAST-{Path(last_model_path).name}"
    artifact.add_file(last_model_path, name=name_last_ckpt)
    logger.info(
        f"Uploading best and last ckpts to wandb: {name_best_ckpt} and {name_last_ckpt} ..."
    )
    if wandb_run is not None:
        wandb_run.log_artifact(artifact)
    else:
        try:
            wandb.log_artifact(artifact)
        except Exception as e:
            logger.error(f"Failed to upload ckpts to wandb: {e}")
    logger.info("Uploading best and last ckpts to wandb ... Done")


def log_config_as_artifact(config: DictConfig, wandb_run: Optional[Run] = None) -> None:
    """
    Log the hydra config as an artifact to wandb.
    """
    try:
        config_dict = OmegaConf.to_container(config, resolve=True)
    except Exception as e:
        logger.error(f"Error converting config to dict: {e}")
        raise e

    # create an artifact
    artifact = wandb.Artifact(name="config", type="config", metadata=config_dict)
    logger.debug(f"Created artifact: {artifact}")

    # save as a temporary file
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        # save config to file
        OmegaConf.save(config=config_dict, f=f.name)
        # add to artifact
        artifact.add_file(f.name)
        logger.debug(f"Added config file to artifact: {f.name}")
        # log as artifact
        if wandb_run is not None:
            wandb_run.log_artifact(artifact)
            logger.debug("Logged config artifact to wandb")
        else:
            try:
                wandb.log_artifact(artifact)
                logger.debug("Logged config artifact to wandb")
            except Exception as e:
                logger.error(f"Failed to upload config to wandb: {e}")


def log_plot_as_artifact(plot_path: Path, wandb_run: Optional[Run] = None) -> None:
    """
    Log the plot as an artifact to wandb.
    """
    artifact = wandb.Artifact(name="correlation-plot", type="plot")
    artifact.add_file(plot_path)
    logger.debug(f"Uploading plot to wandb: {plot_path} ...")
    if wandb_run is not None:
        wandb_run.log_artifact(artifact)
        logger.debug("Logged plot artifact to wandb")
    else:
        try:
            wandb.log_artifact(artifact)
            logger.debug("Logged plot artifact to wandb")
        except Exception as e:
            logger.error(f"Failed to upload plot to wandb: {e}")


# write the local trainer default_root_dir to a file and log as an artifact
def log_default_root_dir(default_root_dir: Union[Path, str], wandb_run: Optional[Run]=None) -> None:
    if isinstance(default_root_dir, str):
        default_root_dir = Path(default_root_dir)
    logger.info(f"Logging as an artifact the default_root_dir: {default_root_dir} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with open(tmpdir / "default_root_dir.txt", "w") as f:
            f.write(default_root_dir.as_posix())
        artifact = wandb.Artifact(
            name="default_root_dir",
            type="file",
        )
        artifact.add_file((tmpdir / "default_root_dir.txt").as_posix())
        if wandb_run is not None:
            wandb_run.log_artifact(artifact)
            logger.debug("Logged default_root_dir artifact to wandb")
        else:
            try:
                wandb.log_artifact(artifact)
                logger.debug("Logged default_root_dir artifact to wandb")
            except Exception as e:
                logger.error(f"Failed to upload default_root_dir to wandb: {e}")


def log_run_dir(run_dir: Union[str, Path], wandb_run: Optional[Run]=None) -> None:
    """
    Log the run directory as an artifact to wandb.
    """
    logger.info(f"Logging as an artifact the run directory: {run_dir} ...")
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # create a tar.gz file of the run_dir
        logger.info(f"Creating a tar.gz file of the run directory: {run_dir} ...")
        with tarfile.open(tmpdir / "run_dir.tar.gz", "w:gz") as tar:
            tar.add(run_dir.as_posix(), arcname=run_dir.name)
        # create an artifact
        artifact = wandb.Artifact(name="run_dir", type="file")
        artifact.add_file((tmpdir / "run_dir.tar.gz").as_posix())
        if wandb_run is not None:
            wandb_run.log_artifact(artifact)
            logger.debug("Logged run_dir artifact to wandb")
        else:
            try:
                wandb.log_artifact(artifact)
                logger.debug("Logged run_dir artifact to wandb")
            except Exception as e:
                logger.error(f"Failed to upload run_dir to wandb: {e}")
