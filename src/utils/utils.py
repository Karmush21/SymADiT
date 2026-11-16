"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig
from src.utils import pylogger, rich_utils


import numpy as np
import torch
import torch.nn.functional as F

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def as_1d_tensor(x, like: torch.Tensor) -> torch.Tensor:
    """
    Convert x (torch/np/list/float) to a 1D torch tensor on same device/dtype as `like`.
    """
    if torch.is_tensor(x):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        # Handles list of floats or list containing an ndarray
        if len(x) == 0:
            t = torch.empty((0,), dtype=like.dtype)
        elif len(x) == 1 and isinstance(x[0], np.ndarray):
            t = torch.from_numpy(x[0])
        else:
            t = torch.tensor(x)
    else:
        # scalar
        t = torch.tensor([x])

    t = t.to(device=like.device, dtype=like.dtype)
    return t.reshape(-1)

def safe_mse_1d(pred: torch.Tensor, target, *, treat_empty_as_zero=True) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = as_1d_tensor(target, like=pred)

    # Empty handling
    if pred.numel() == 0 and target.numel() == 0:
        return pred.new_tensor(0.0) if treat_empty_as_zero else pred.new_tensor(float("nan"))

    # If one is empty but not the other, that's a bug
    if pred.numel() != target.numel():
        raise RuntimeError(f"free-vector length mismatch: pred {pred.numel()} vs target {target.numel()}")

    return F.mse_loss(pred, target, reduction="mean")

def safe_mae_1d(pred: torch.Tensor, target, *, treat_empty_as_zero=True) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = as_1d_tensor(target, like=pred)

    if pred.numel() == 0 and target.numel() == 0:
        return pred.new_tensor(0.0) if treat_empty_as_zero else pred.new_tensor(float("nan"))

    if pred.numel() != target.numel():
        raise RuntimeError(f"free-vector length mismatch: pred {pred.numel()} vs target {target.numel()}")

    return (pred - target).abs().mean()


def safe_mae_1d_periodic(pred: torch.Tensor, target, *, period=1.0, treat_empty_as_zero=True) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = as_1d_tensor(target, like=pred)

    if pred.numel() == 0 and target.numel() == 0:
        return pred.new_tensor(0.0) if treat_empty_as_zero else pred.new_tensor(float("nan"))

    if pred.numel() != target.numel():
        raise RuntimeError(f"free-vector length mismatch: pred {pred.numel()} vs target {target.numel()}")

    delta = pred - target
    delta = torch.remainder(delta + 0.5 * period, period) - 0.5 * period

    return delta.abs().mean()