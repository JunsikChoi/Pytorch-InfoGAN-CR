import torch
from utils import save_config
from trainer import Trainer
from config import get_config
from data_loader import get_loader


def main(config):
    save_config(config)
    data_loader = get_loader(
        config.batch_size, config.project_root, config.dataset)
    trainer = Trainer(config, data_loader)
    trainer.train()
    return


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
