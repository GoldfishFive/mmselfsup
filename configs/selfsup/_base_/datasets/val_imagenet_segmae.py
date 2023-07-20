# dataset settings
dataset_type = 'ImageNet'
data_root = '/home/user01/datasets/imagenet_val/'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='RandomResizedCrop',
#         size=224,
#         scale=(0.2, 1.0),
#         backend='pillow',
#         interpolation='bicubic'),
#     dict(type='RandomFlip', prob=0.5),# plan to give up
#     dict(type='SegMAEMaskGenerator',
#          sigma=0.8,
#          kernel=11,
#          num_of_partitions=500,
#          min_area_size=500),
#     dict(type='PackSelfSupInputs', meta_keys=['img_path','seg_mask', 'num_of_objects'])
# ]

# train_dataloader = dict(
#     batch_size=128,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     collate_fn=dict(type='default_collate'),
#     # 训练数据集配置
#     dataset=dict(
#         type='ImageList',
#         # type='CustomDataset',
#         ann_file='',
#         data_prefix=data_root,
#         # with_label=False,  # 对于无监督任务，使用 False
#         pipeline=train_pipeline
#     )
# )
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFHSegMap'),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),# plan to give up
    dict(type='PackSelfSupInputs', meta_keys=['img_path','gt_seg_mask', 'num_of_objects'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    # 训练数据集配置
    dataset=dict(
        type='SegMAEImageList',
        ann_file='',
        ann_root='/home/data1/group1/datasets/imagenet_val_FH_500_500/',
        data_prefix=data_root,
        pipeline=train_pipeline
    )
)



