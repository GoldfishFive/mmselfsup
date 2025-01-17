_base_ = [
    '../../_base_/models/mae_vit-base-p16.py',
    '../../_base_/datasets/val_imagenet_mae.py',
    '../../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../../_base_/default_runtime.py',
]

# dataset 8 x 128
train_dataloader = dict(batch_size=128, num_workers=16)
total_batch = 128
# optimizer wrapper
optimizer = dict(
    type='AdamW', lr=1.5e-4 * total_batch / 256, betas=(0.9, 0.95), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=270,
        by_epoch=True,
        begin=30,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
# pre-train for 300 epochs
train_cfg = dict(max_epochs=300)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=30))


# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True

