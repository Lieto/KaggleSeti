from libraries import *
from configuration import CFG


def get_transforms(*, data, cfg: CFG):

    if data == "train":

        return A.Compose([
            A.Resize(cfg.size, cfg.size),
            ToTensorV2(),
        ])
    elif data == "valid":

        return A.Compose([
            A.Resize(cfg.size, cfg.size),
            ToTensorV2(),
        ])
