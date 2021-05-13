import time

from libraries import *

from loguru import logger
from helpers import AverageMeter, time_since


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg):

    if cfg.apex:
        scaler = GradScaler()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()
    start = end = time.time()
    global_step = 0

    for step, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        if cfg.apex:
            with autocast():
                y_preds = model(images)
                loss = criterion(y_preds.view(-1), labels)
        else:
            y_preds = model(images)
            loss = criterion(y_preds.view(-1), labels)

        losses.update(loss.item(), batch_size)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        if cfg.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        if (step + 1) % cfg.gradient_accumulation_steps == 0:

            if cfg.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()

        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):

            logger.info(f"Epoch: [{epoch+1}][{step}/{len(train_loader)}]")
            logger.info(f"Data: {data_time.val:.3f} ({data_time.avg:.3f})")
            logger.info(f"Elapsed: {time_since(start, float(step+1)/len(train_loader))}")
            logger.info(f"Loss: {losses.val:.4f} (){losses.avg:.4f}")
            logger.info(f"Grad: {grad_norm:.4f}")

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, cfg):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.eval()

    preds = []
    start = end = time.time()

    for step, (images, labels) in enumerate(valid_loader):

        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images)

        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)

        preds.append(y_preds.sigmoid().to("cpu").numpy())

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        batch_time.update(time.time() - end)

        end = time.time()

        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):

            logger.info(f"EVAL: [{step}/{len(valid_loader)}]")
            logger.info(f"Data: {data_time.val:.3f} ({data_time.avg:.3f})")
            logger.info(f"elapsed: {time_since(start, float(step+1)/len(valid_loader))}")
            logger.info(f"Loss: {losses.val:.4f} ({losses.avg:.4f})")

    predictions = np.concatenate(preds)

    return losses.avg, predictions


