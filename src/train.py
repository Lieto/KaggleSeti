import torch

from libraries import *
from loguru import logger
from typing import List
from configuration import CFG
from dataset import TrainDataset
from transforms import get_transforms
from model import CustomModel
from utils import get_score
from train_functions import train_fn, valid_fn


def train_loop(folds: pd.DataFrame, fold, cfg: CFG):

    logger.info(f"========= fold: {fold} training =============")

    # ==================
    # loader
    # ==================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    folds = cfg.train_df

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[cfg.target_col].values

    train_dataset = TrainDataset(cfg=cfg, transform=get_transforms(data="train", cfg=cfg))
    valid_dataset = TrainDataset(cfg=cfg, transform=get_transforms(data="valid", cfg=cfg))

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size * 2,
                              shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # ================================
    # scheduler
    # =================================

    def get_scheduler(optimizer):
        if cfg.scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.factor, patience=cfg.patience, verbose=True, eps=cfg.eps)
        elif cfg.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr, last_epoch=-1)
        elif cfg.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_==cfg.T_0, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1)

        return scheduler

    # =================================
    # model & optimizer
    # ===================================
    model = CustomModel(cfg, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # =======================================
    # loop
    # ============================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0
    best_loss = np.inf

    for epoch in range(cfg.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device, cfg)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        logger.info(f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f} avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s")
        logger.info(f"epoch {epoch+1} - Score: {score:.4f}")

        if score > best_score:
            best_score = score

            logger.info(f"Epoch {epoch+1} - Save best score: {best_score:.4f} Model")
            torch.save({
                "model": model.state_dict(),
                "preds": preds
            },
            "data/output/" + f"{cfg.model_name}_fold{fold}_best_score.pth")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

            logger.info(f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model")
            torch.save({
                "model": model.state_dict(),
                "preds": preds,
            },
            "data/output/" + f"{cfg.model_name}_fold{fold}_best_loss.pth")

    valid_folds["preds"] = torch.load("data/output/" + f"{cfg.model_name}_fold{fold}_best_loss.pth",
                                      map_location=torch.device("cpu"))["preds"]

    return valid_folds
