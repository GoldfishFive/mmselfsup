_base_ = [
    '../_base_/datasets/val_imagenet_segmae_fh.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]
# model settings
model = dict(
    type='SegMAE',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='SegMAEViT',
        arch='small',
        patch_size=16,
        mask_ratio=0.50,
        fix_mask_ratio=True,# True used the fixed mask_ratio 0.75 during training;
        max_epochs=300, # when fix_mask_ratio is False, mask_ratio change from low_mask_ratio to high_mask_ratio
        low_mask_ratio=0.35,
        high_mask_ratio=0.85
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='MAEReconstructionLoss')),
    init_cfg=[
        dict(type='Xavier', distribution='uniform', layer='Linear'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

# dataset 8 x 128
train_dataloader = dict(batch_size=96, num_workers=16)
# total_batch = 4096
# total_batch = 1024
# total_batch = 256
# total_batch = 768
#total_batch = 384
total_batch = 96
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
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='SetEpochHook',
        start_epoch=0
)]
# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True
