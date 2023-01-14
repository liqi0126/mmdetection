_base_ = ['./mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_mask_only.py']

load_from = 'checkpoints/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth'

data_root = 'data/coco/'

model = dict(
    clip_only=True,
    panoptic_head=dict(
        loss_clip=dict(
            type='MSELoss',
            reduction='none',
        ),
        loss_mask=None,
        loss_dice=None,
    ),
)

image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        # label_key='category',
        label_key='clip_feat',
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_size=image_size,
    #     crop_type='absolute',
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

data = dict(
    train=dict(
        # ann_file=data_root + 'annotations/panoptic_train2017.json',
        ann_file=data_root + 'annotations/panoptic_clip_train2017.json',
        pipeline=train_pipeline,
        ),
    val=dict(
        # ann_file=data_root + 'annotations/panoptic_val2017.json',
        ann_file=data_root + 'annotations/panoptic_clip_val2017.json',
        ),
    test=dict(
        # ann_file=data_root + 'annotations/panoptic_val2017.json',
        ann_file=data_root + 'annotations/panoptic_clip_val2017.json',
        )
)
