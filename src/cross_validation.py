from libraries import *
from loguru import logger
from configuration import CFG


def cv_split(cfg: CFG, train: pd.DataFrame):

    fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=CFG.seed)

    for n, (train_index, val_index) in enumerate(fold.split(train, train[cfg.target_col])):
        train.loc[val_index, "fold"] = int(n)

    train["fold"] = train["fold"].astype(int)

    logger.info(f"Train group by fold target: {train.groupby(['fold', 'target']).size()}")

    return train

