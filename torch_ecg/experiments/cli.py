import hydra
from omegaconf import DictConfig, OmegaConf

from torch_ecg_volta.experiments.manager import ExperimentManager


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    with ExperimentManager(cfg):
        pass


if __name__ == "__main__":
    main()
