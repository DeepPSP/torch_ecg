from contextlib import ExitStack

import mlflow
from omegaconf import DictConfig, OmegaConf


class ExperimentManager(ExitStack):
    """A context manager class to perform basic environment control on MLflow, the configuration, etc."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        if not hasattr(config, "experiment") or not isinstance(config.experiment, str):
            raise ValueError("Your configuration should contain an `experiment` field!")

    def __enter__(self):
        mlflow.set_experiment(self.config.experiment)
        self.enter_context(mlflow.start_run())
        mlflow.log_dict(OmegaConf.to_container(self.config), "config.yaml")
