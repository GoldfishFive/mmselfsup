# dataset settings
dataset_type = 'ImageNet'
data_root = '/home/user01/datasets/imagenet_val/'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    # dataset=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     # ann_file='meta/train.txt',
    #     ann_file='words.txt',
    #     data_prefix=dict(img_path='train/'),
    #     pipeline=train_pipeline)
    # 训练数据集配置
    dataset=dict(
        type='ImageList',
        # type='CustomDataset',
        ann_file='',
        data_prefix=data_root,
        # with_label=False,  # 对于无监督任务，使用 False
        pipeline=train_pipeline
    )

)
