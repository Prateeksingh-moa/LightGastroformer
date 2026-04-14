# config.py - Central configuration for LightGastroFormer

class Config:
    # Paths 
    data_dir  = '/path/to/the/dataset'
    csv_file  = '/path/to/the/csv'
    save_path = 'best_model.pth'

    # Data 
    img_size    = 224
    split_ratio = 0.8
    batch_size  = 16
    num_workers = 2

    # Model 
    patch_size      = 8
    embed_dim       = 256
    depth           = 6
    num_heads       = 4
    mlp_ratio       = 2.0
    drop_rate       = 0.1
    attn_drop_rate  = 0.1
    qkv_bias        = True

    # Training 
    epochs        = 15
    lr            = 1e-4
    weight_decay  = 1e-2
    eta_min       = 1e-6
    aux_loss_weight = 0.4

    # Misc 
    seed = 42