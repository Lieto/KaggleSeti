from libraries import *
from configuration import CFG
from loguru import logger


class CustomModel(nn.Module):

    def __init__(self, cfg: CFG, pretrained: bool = False):

        super().__init__()
        self.cfg = cfg
        #logger.info(f"Available model: {timm.models.list_models()}")
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained, in_chans=1)
        self.n_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(self.n_features, self.cfg.target_size)

        logger.info(f"Model n_features: {self.n_features}")
        logger.info(f"Model head.fc: {self.model.head.fc}")

    def forward(self, x):

        output = self.model(x)
        return output
