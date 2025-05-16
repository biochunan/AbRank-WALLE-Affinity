"""
Contains project-level constants used to configure paths and wandb logging.

Paths are configured using the `.env` file in the project root.
"""
import logging
import os
from pathlib import Path

import rootutils
from loguru import logger

# ---------------- PATH CONSTANTS ----------------
PROJECT_PATH = rootutils.find_root(search_from=__file__, indicator=".project-root")
SRC_PATH = Path(__file__).parent
HYDRA_CONFIG_PATH = SRC_PATH / "config"

logger.info(f"PROJECT_PATH: {PROJECT_PATH}")
logger.info(f"SRC_PATH: {SRC_PATH}")
logger.info(f"HYDRA_CONFIG_PATH: {HYDRA_CONFIG_PATH}")

# ---------------- ENVIRONMENT VARIABLES ----------------
# including:
#   ROOT_DIR
#   RUNS_PATH
#   DATA_PATH
# Data paths are configured using the `.env` file in the project root.

if not (PROJECT_PATH / ".env").exists():
    logger.debug("No `.env` file found in project root. Checking for env vars...")
    # If no `.env` file found, check for an env var
    if os.environ.get("DATA_PATH") is not None:
        logger.debug("Found env var `DATA_PATH`:.")
        DATA_PATH = os.environ.get("DATA_PATH")
    else:
        logger.debug("No env var `DATA_PATH` found. Setting default...")
        DATA_PATH = str(SRC_PATH / "data")
        os.environ["DATA_PATH"] = str(DATA_PATH)
else:
    import dotenv  # lazy import to avoid dependency on dotenv

    dotenv.load_dotenv(
        dotenv_path=PROJECT_PATH / ".env",
        override=True,  # overwrite os env paths using .env file
    )
    logger.debug("Loaded .env file")

# Set default environment paths as fallback if not specified in .env file
#  NOTE: These will be overridden by paths in the hydra config or by
#   the corresponding `.env` environment variables if they are set.
#   We provide them simply as a fallback for users who do not want to
#   use hydra or environment variables.
if os.environ.get("ROOT_DIR") is None:
    ROOT_DIR = str(PROJECT_PATH)
    os.environ["ROOT_DIR"] = str(ROOT_DIR)
else:
    ROOT_DIR = os.environ.get("ROOT_DIR")
if os.environ.get("DATA_PATH") is None:
    DATA_PATH = str(PROJECT_PATH / "data")
    os.environ["DATA_PATH"] = str(DATA_PATH)
else:
    DATA_PATH = os.environ.get("DATA_PATH")
if os.environ.get("RUNS_PATH") is None:
    RUNS_PATH = str(PROJECT_PATH / "runs")
    os.environ["RUNS_PATH"] = str(RUNS_PATH)
else:
    RUNS_PATH = os.environ.get("RUNS_PATH")
if os.environ.get("PROJECT_DIR") is None:
    PROJECT_DIR = str(PROJECT_PATH)
    os.environ["PROJECT_DIR"] = str(PROJECT_DIR)
else:
    PROJECT_DIR = os.environ.get("PROJECT_DIR")

logger.info(f"ROOT_DIR : {ROOT_DIR}")
logger.info(f"RUNS_PATH: {RUNS_PATH}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"PROJECT_DIR: {PROJECT_DIR}")

# ---------------- WANDB CONSTANTS ----------------
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
"""API key for wandb."""

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
"""Entity for wandb logging."""

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
"""Project name for wandb logging."""

# ---------------- LOGGING CONSTANTS ----------------
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s [in %(funcName)s at %(pathname)s:%(lineno)d]"
)
DEFAULT_LOG_FILE = PROJECT_PATH / "logs" / "default_log.log"
DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_LEVEL = logging.DEBUG  # verbose logging per default
