"""
loggers, including (planned) CSVLogger, TensorBoardXLogger, WandbLogger

with reference to `loggers` of `textattack` and `loggers` of `pytorch-lightning`

"""

import csv
import logging
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import tensorboardX
import torch

from ..cfg import DEFAULTS
from ..utils.misc import ReprMixin, add_docstring, get_date_str, init_logger

__all__ = [
    "BaseLogger",
    "TxtLogger",
    "CSVLogger",
    "TensorBoardXLogger",
    # "WandbLogger",
    "LoggerManager",
]


_log_metrics_doc = """Log metrics.

        Parameters
        ----------
        metrics : dict
            The metrics to be logged.
        step : int, optional
            The number of (global) steps of training.
        epoch: int, optional,
            The epoch number of training.
        part : str, optional
            The part of the training data the metrics computed from,
            can be "train" or "val" or "test", etc.

        """

_log_message_doc = """Log a message.

        Parameters
        ----------
        msg : str
            The message to be logged.
        level : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}, optional
            The level of the message.

        """

_epoch_start_doc = """Message logged at the start of each epoch.

        Parameters
        ----------
        epoch : int
            The number of the epoch.

        """

_epoch_end_doc = """Actions to be performed at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The number of the epoch.

        """


class BaseLogger(ReprMixin, ABC):
    """The abstract base class of all loggers."""

    __name__ = "BaseLogger"

    @add_docstring(_log_metrics_doc)
    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "train",
    ) -> None:
        raise NotImplementedError

    @add_docstring(_log_message_doc)
    @abstractmethod
    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """Flush the logger."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> Any:
        """Initialize the logger from a config dict."""
        raise NotImplementedError

    @add_docstring(_epoch_start_doc)
    def epoch_start(self, epoch: int) -> None:
        pass

    @add_docstring(_epoch_end_doc)
    def epoch_end(self, epoch: int) -> None:
        pass

    @property
    def log_dir(self) -> str:
        """Directory to save the log file."""
        return self._log_dir

    @property
    @abstractmethod
    def filename(self) -> str:
        """Filename of the log file."""
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "filename",
        ]


class TxtLogger(BaseLogger):
    """Logger that logs to a text file.

    Parameters
    ----------
    log_dir : `path-like`, optional
        The directory to save the log file.
    log_suffix : str, optional
        The suffix of the log file.

    """

    __name__ = "TxtLogger"

    def __init__(
        self,
        log_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        log_suffix: Optional[str] = None,
    ) -> None:
        self._log_dir = Path(log_dir or DEFAULTS.log_dir)
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._W_OK = os.access(self._log_dir, os.W_OK)
        except PermissionError:
            warnings.warn(f"Directory {self._log_dir} is not writable.")
            self._W_OK = False
            return  # do not create the logger if the directory is not writable
        if log_suffix is None:
            log_suffix = ""
        else:
            log_suffix = f"_{log_suffix}"
        self.log_file = f"{DEFAULTS.prefix}_{get_date_str()}{log_suffix}.txt"
        self.logger = init_logger(self.log_dir, self.log_file, verbose=1)
        self.step = -1

    @add_docstring(_log_metrics_doc)
    def log_metrics(
        self,
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "train",
    ) -> None:
        if not self._W_OK:
            return
        if step is not None:
            self.step = step
        else:
            self.step += 1
        prefix = f"Step {step}: "
        if epoch is not None:
            prefix = f"Epoch {epoch} / {prefix}"
        _metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        spaces = len(max(_metrics.keys(), key=len))
        msg = (
            f"{part.capitalize()} Metrics:\n{self.short_sep}\n"
            + "\n".join([f"{prefix}{part}/{k} : {' '*(spaces-len(k))}{v:.4f}" for k, v in _metrics.items()])
            + f"\n{self.short_sep}"
        )
        self.log_message(msg)

    @add_docstring(_log_message_doc)
    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        if not self._W_OK:
            return
        self.logger.log(level, msg)

    @property
    def long_sep(self) -> str:
        return "-" * 110

    @property
    def short_sep(self) -> str:
        return "-" * 50

    @add_docstring(_epoch_start_doc)
    def epoch_start(self, epoch: int) -> None:
        if not self._W_OK:
            return
        self.logger.info(f"Train epoch_{epoch}:\n{self.long_sep}")

    @add_docstring(_epoch_end_doc)
    def epoch_end(self, epoch: int) -> None:
        if not self._W_OK:
            return
        self.logger.info(f"{self.long_sep}\n")

    def flush(self) -> None:
        """Flush the log file."""
        if not self._W_OK:
            return
        for h in self.logger.handlers:
            if hasattr(h, "flush"):
                h.flush()

    def close(self) -> None:
        """Close the log file."""
        if not self._W_OK:
            logging.shutdown()
            return
        for h in self.logger.handlers:
            h.close()
            self.logger.removeHandler(h)
        logging.shutdown()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TxtLogger":
        """Create a logger from a config.

        Parameters
        ----------
        config : dict
            The config to create the logger.

        Returns
        -------
        TxtLogger
            The instance of the created logger.

        """
        return cls(config.get("log_dir", None), config.get("log_suffix", None))

    @property
    def filename(self) -> str:
        """Filename of the log file."""
        if not self._W_OK:
            return ""
        return str(self.log_dir / self.log_file)


class CSVLogger(BaseLogger):
    """Logger that logs to a CSV file.

    Parameters
    ----------
    log_dir : `path-like`, optional
        The directory to save the log file.
    log_suffix : str, optional
        The suffix of the log file.

    """

    __name__ = "CSVLogger"

    def __init__(
        self,
        log_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        log_suffix: Optional[str] = None,
    ) -> None:
        self._log_dir = Path(log_dir or DEFAULTS.log_dir)
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._W_OK = os.access(self._log_dir, os.W_OK)
        except PermissionError:
            warnings.warn(f"Directory {self._log_dir} is not writable.")
            self._W_OK = False
            return
        if log_suffix is None:
            log_suffix = ""
        else:
            log_suffix = f"_{log_suffix}"
        self.log_file = f"{DEFAULTS.prefix}_{get_date_str()}{log_suffix}.csv"
        self.logger = pd.DataFrame()
        self.step = -1
        self._flushed = True

    @add_docstring(_log_metrics_doc)
    def log_metrics(
        self,
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "train",
    ) -> None:
        if not self._W_OK:
            return
        if step is not None:
            self.step = step
        else:
            self.step += 1
        row = {"step": self.step, "time": datetime.now(), "part": part}
        if epoch is not None:
            row.update({"epoch": epoch})
        row.update({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()})
        # self.logger = self.logger.append(row, ignore_index=True)
        self.logger = pd.concat([self.logger, pd.DataFrame(row, index=[0])], ignore_index=True)

        self._flushed = False

    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        pass

    def flush(self) -> None:
        """Flush the log file."""
        if not self._W_OK:
            self._flushed = True
            return
        if not self._flushed:
            self.logger.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
            self._flushed = True

    def close(self) -> None:
        """Close the log file."""
        self.flush()

    def __del__(self):
        self.flush()
        del self

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CSVLogger":
        """Create a logger from a config.

        Parameters
        ----------
        config : dict
            The config to create the logger.

        Returns
        -------
        CSVLogger
            The instance of the created logger.

        """
        return cls(config.get("log_dir", None), config.get("log_suffix", None))

    @property
    def filename(self) -> str:
        """Filename of the log file."""
        if not self._W_OK:
            return ""
        return str(self.log_dir / self.log_file)


class TensorBoardXLogger(BaseLogger):
    """Logger that logs to a TensorBoardX file.

    Parameters
    ----------
    log_dir : `path-like`, optional
        The directory to save the log file.
    log_suffix : str, optional
        The suffix of the log file.

    """

    __name__ = "TensorBoardXLogger"

    def __init__(
        self,
        log_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        log_suffix: Optional[str] = None,
    ) -> None:
        self._log_dir = Path(log_dir or DEFAULTS.log_dir)
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._W_OK = os.access(self._log_dir, os.W_OK)
        except PermissionError:
            warnings.warn(f"Directory {self._log_dir} is not writable.")
            self._W_OK = False
            return
        self.logger = tensorboardX.SummaryWriter(str(self._log_dir), filename_suffix=log_suffix or "")
        self.log_file = self.logger.file_writer.event_writer._ev_writer._file_name
        self.step = -1

    @add_docstring(_log_metrics_doc)
    def log_metrics(
        self,
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "train",
    ) -> None:
        if not self._W_OK:
            return
        if step is not None:
            self.step = step
        else:
            self.step += 1
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.logger.add_scalar(f"{part}/{k}", v, self.step)

    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        pass

    def flush(self) -> None:
        """Flush the log file."""
        if not self._W_OK:
            return
        self.logger.flush()

    def close(self) -> None:
        """Close the log file."""
        if not self._W_OK:
            return
        self.logger.close()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TensorBoardXLogger":
        """Create a logger from a config.

        Parameters
        ----------
        config : dict
            The config to create the logger.

        Returns
        -------
        TensorBoardXLogger
            The instance of the created logger.

        """
        return cls(config.get("log_dir", None), config.get("log_suffix", None))

    @property
    def filename(self) -> str:
        """Filename of the log file."""
        return str(self.log_dir / self.log_file)


# class WandbLogger(BaseLogger):
#     """ """

#     __name__ = "WandbLogger"

#     def __init__(
#         self,
#         log_dir: Optional[Union[str, bytes, os.PathLike]] = None,
#         log_suffix: Optional[str] = None,
#         project: Optional[str] = None,
#         entity: Optional[str] = None,
#         hyperparameters: Optional[dict] = None,
#     ) -> None:
#         """
#         to write docstring
#         """
#         self.__wandb = importlib.import_module("wandb")
#         self._log_dir = Path(log_dir or DEFAULTS.log_dir)
#         self._log_dir.mkdir(parents=True, exist_ok=True)
#         self._log_suffix = log_suffix
#         self._project = project
#         self._entity = entity
#         self._hyperparameters = hyperparameters
#         self.__wandb.init(
#             name=self._log_suffix,
#             dir=self._log_dir,
#             config=self._hyperparameters,
#             project=self._project,
#             entity=self._entity,
#         )

#     def log_metrics(
#         self, metrics: Dict[str, float], step: Optional[int] = None
#     ) -> None:
#         """ """
#         self.__wandb.log(metrics, step=step)

#     def log_message(self, msg: str, level: int = logging.INFO) -> None:
#         pass

#     def flush(self) -> None:
#         """ """
#         pass

#     def close(self) -> None:
#         """ """
#         self.__wandb.finish()

#     @classmethod
#     def from_config(cls, config: Dict[str, Any]) -> "WandbLogger":
#         """ """
#         return cls(
#             config.get("log_dir", None),
#             config.get("log_suffix", None),
#             config.get("project", None),
#             config.get("entity", None),
#             config.get("hyperparameters", None),
#         )

#     @property
#     def filename(self) -> str:
#         """ """
#         return self.__wandb.run.dir


class LoggerManager(ReprMixin):
    """Manager of loggers.

    This class manages multiple loggers and provides a unified interface to
    log metrics and messages.

    Parameters
    ----------
    log_dir : `path-like`, optional
        The directory to save the log file.
    log_suffix : str, optional
        The suffix of the log file.

    """

    __name__ = "LoggerManager"

    def __init__(
        self,
        log_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        log_suffix: Optional[str] = None,
    ) -> None:
        self._log_dir = Path(log_dir or DEFAULTS.log_dir)
        self._log_suffix = log_suffix
        self._loggers = []

    def _add_txt_logger(self) -> None:
        """Add a text logger to the manager."""
        self.loggers.append(TxtLogger(self._log_dir, self._log_suffix))

    def _add_csv_logger(self) -> None:
        """Add a csv logger to the manager."""
        self.loggers.append(CSVLogger(self._log_dir, self._log_suffix))

    def _add_tensorboardx_logger(self) -> None:
        """Add a tensorboardx logger to the manager."""
        self.loggers.append(TensorBoardXLogger(self._log_dir, self._log_suffix))

    # def _add_wandb_logger(self, **kwargs: dict) -> None:
    #     """ """
    #     raise NotImplementedError("NOT tested yet!")
    # self.loggers.append(WandbLogger(self._log_dir, self._log_suffix, **kwargs))

    @add_docstring(_log_metrics_doc)
    def log_metrics(
        self,
        metrics: Dict[str, Union[Real, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        part: str = "train",
    ) -> None:
        for lgs in self.loggers:
            lgs.log_metrics(metrics, step, epoch, part)

    @add_docstring(_log_message_doc)
    def log_message(self, msg: str, level: int = logging.INFO) -> None:
        for lgs in self.loggers:
            lgs.log_message(msg, level)

    @add_docstring(_epoch_start_doc)
    def epoch_start(self, epoch: int) -> None:
        for lgs in self.loggers:
            lgs.epoch_start(epoch)

    @add_docstring(_epoch_end_doc)
    def epoch_end(self, epoch: int) -> None:
        for lgs in self.loggers:
            lgs.epoch_end(epoch)

    def flush(self) -> None:
        """Flush the loggers."""
        for lgs in self.loggers:
            lgs.flush()

    def close(self) -> None:
        """Close the loggers."""
        for lgs in self.loggers:
            lgs.close()

    @property
    def loggers(self) -> List[BaseLogger]:
        """List of loggers."""
        return self._loggers

    @property
    def log_dir(self) -> str:
        """Directory to save the log file."""
        return self._log_dir

    @property
    def log_suffix(self) -> str:
        """Suffix of the log file."""
        return self._log_suffix

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoggerManager":
        """Create a logger manager from a configuration.

        Parameters
        ----------
        config : dict
            The configuration of the logger manager.

        Returns
        -------
        LoggerManager
            The instance of the created logger manager.

        """
        lm = cls(config.get("log_dir", None), config.get("log_suffix", None))
        if config.get("txt_logger", True):
            lm._add_txt_logger()
        if config.get("csv_logger", True):
            lm._add_csv_logger()
        if config.get("tensorboardx_logger", True):
            lm._add_tensorboardx_logger()
        # if config.get("wandb_logger", False):
        #     kwargs = config.get("wandb_logger", {})
        #     lm._add_wandb_logger(**kwargs)
        return lm

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "loggers",
        ]
