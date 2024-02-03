from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg, resolve=False))
    print(OmegaConf.to_yaml(cfg, resolve=True))

if __name__ == "__main__":
    my_app()
