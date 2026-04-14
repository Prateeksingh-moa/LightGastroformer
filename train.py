#!/usr/bin/env python3
# train.py — Entry point for LightGastroFormer training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix

from config  import Config
from dataset import KvasirCapsuleDataset, get_transforms
from model   import LightGastroFormer
from engine  import train_model, validate
from utils   import set_seed, get_logger, plot_metrics, plot_confusion_matrix


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Transforms 
    train_tf, val_tf = get_transforms(cfg.img_size)

    # Datasets 
    train_ds = KvasirCapsuleDataset(
        cfg.data_dir, cfg.csv_file,
        transform=train_tf, split='train',
        split_ratio=cfg.split_ratio, seed=cfg.seed,
    )
    val_ds = KvasirCapsuleDataset(
        cfg.data_dir, cfg.csv_file,
        transform=val_tf, split='val',
        split_ratio=cfg.split_ratio, seed=cfg.seed,
    )

    # Data loaders 
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
    )

    logger.info(f"Classes : {train_ds.classes}")
    logger.info(f"Train   : {len(train_ds)} samples")
    logger.info(f"Val     : {len(val_ds)} samples")

    # Model 
    model = LightGastroFormer(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        num_classes=len(train_ds.classes),
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Optimisation 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)

    # Training 
    logger.info("Starting training …")
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        num_epochs=cfg.epochs,
        device=device,
        save_path=cfg.save_path,
        aux_weight=cfg.aux_loss_weight,
    )
    plot_metrics(history)

    # Final evaluation 
    logger.info("Loading best checkpoint for final evaluation …")
    checkpoint = torch.load(cfg.save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_acc, preds, gt = validate(model, val_loader, criterion, device)
    logger.info(f"Final validation accuracy: {val_acc:.4f}")

    cm = confusion_matrix(gt, preds)
    plot_confusion_matrix(cm, train_ds.classes)

    logger.info("Done.")


if __name__ == '__main__':
    main()