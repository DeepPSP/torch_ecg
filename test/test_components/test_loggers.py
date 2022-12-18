"""
"""

from pathlib import Path

import pytest
import torch

from torch_ecg.components.loggers import (
    BaseLogger,
    TxtLogger,
    CSVLogger,
    TensorBoardXLogger,
    LoggerManager,
)


_LOG_DIR = Path(__file__).parents[1] / "logs"


def test_base_logger():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseLogger()


def test_logger_manager():
    config = {
        "log_dir": _LOG_DIR,
        "log_suffix": "test",
        "txt_logger": True,
        "csv_logger": True,
        "tensorboardx_logger": True,
    }
    lm = LoggerManager.from_config(config)

    assert lm.log_dir == _LOG_DIR
    assert lm.log_suffix == "test"
    assert str(lm) == repr(lm)

    with pytest.raises(NotImplementedError):
        lm._add_wandb_logger()  # not implemented yet

    lm.log_message("test")
    lm.log_metrics({"test": torch.scalar_tensor(1.1)})

    lm.flush()
    lm.close()


def test_txt_logger():
    config = {
        "log_dir": _LOG_DIR,
        "log_suffix": "test",
    }
    logger = TxtLogger.from_config(config)

    assert logger.log_dir == _LOG_DIR
    assert str(logger) == repr(logger)

    logger.log_message("test")
    logger.log_metrics({"test": torch.scalar_tensor(1.1)})

    logger.flush()
    logger.close()

    assert logger.filename == str(_LOG_DIR / logger.log_file)
    assert Path(logger.filename).exists()
    assert str(logger) == repr(logger)


def test_csv_logger():
    config = {
        "log_dir": _LOG_DIR,
        "log_suffix": "test",
    }
    logger = CSVLogger.from_config(config)

    assert logger.log_dir == _LOG_DIR
    assert str(logger) == repr(logger)

    logger.log_message("test")
    logger.log_metrics({"test": torch.scalar_tensor(1.1)})

    logger.flush()
    logger.close()

    assert logger.filename == str(_LOG_DIR / logger.log_file)
    assert Path(logger.filename).exists()
    assert str(logger) == repr(logger)


def test_tensorboardx_logger():
    config = {
        "log_dir": _LOG_DIR,
        "log_suffix": "test",
    }
    logger = TensorBoardXLogger.from_config(config)

    assert logger.log_dir == _LOG_DIR
    assert str(logger) == repr(logger)

    logger.log_message("test")
    logger.log_metrics({"test": torch.scalar_tensor(1.1)})

    logger.flush()
    logger.close()

    assert logger.filename == str(_LOG_DIR / logger.log_file)
    assert Path(logger.filename).exists()
    assert str(logger) == repr(logger)
