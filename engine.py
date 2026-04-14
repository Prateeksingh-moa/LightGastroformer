# engine.py - Training and validation loops

import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils import get_logger

logger = get_logger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_weight: float = 0.4,
):
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, aux_logits = model(inputs)
        loss = criterion(logits, labels) + aux_weight * criterion(aux_logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc':  f'{accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()):.4f}',
        })

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    aux_weight: float = 0.4,
):
    """Run one validation pass. Returns (loss, accuracy, predictions, ground_truth)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            logits, aux_logits = model(inputs)
            loss = criterion(logits, labels) + aux_weight * criterion(aux_logits, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc':  f'{accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()):.4f}',
            })

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    num_epochs: int,
    device: torch.device,
    save_path: str       = 'best_model.pth',
    aux_weight: float    = 0.4,
) -> dict:
    """
    Full training loop with checkpointing.

    Returns a history dict with keys:
        train_loss, train_acc, val_loss, val_acc
    """
    best_acc = 0.0
    history  = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, aux_weight,
        )
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, aux_weight,
        )
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy':             val_acc,
            }, save_path)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        msg = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        print('\n' + '='*80)
        print(msg)
        print('='*80)
        logger.info(msg)

    return history