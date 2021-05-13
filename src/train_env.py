import pandas as pd

from libraries import *
from loguru import logger
from data_loading import load_data
from configuration import CFG
from model import CustomModel
from utils import seed_torch
from train import train_loop
from utils import get_score

import argparse

def get_result(result_df, cfg):

    preds = result_df["preds"].values
    labels = result_df[cfg.target_col].values
    score = get_score(labels, preds)
    logger.info(f"Score: {score:<.4f}")


def main(args):

    data_dir = args.data_dir

    train_df_example, test_df_example = load_data()

    cfg = CFG(train_df=train_df_example, debug=False)

    logger.info(f"Device: {device}")
    logger.info(f"Train data: {cfg.train_df}")

    seed_torch(cfg.seed)

    cfg.cv_split()

    model = CustomModel(cfg, pretrained=True)

    if cfg.train:

        oof_df = pd.DataFrame()

        for fold in range(cfg.n_fold):
            if fold in cfg.trn_fold:
                _oof_df = train_loop(cfg.train_df, fold, cfg)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"==================== fold: {fold} result ===================")
                get_result(_oof_df)

        logger.info(f"================== CV ==========================")
        get_result(oof_df, cfg)

        oof_df.to_csv("data/output/" + "oof_df.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="")

    args = parser.parse_args()

    main(args)
