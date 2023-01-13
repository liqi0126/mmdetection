_base_ = ['./mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py']

model = dict(
    panoptic_head=dict(
        clip_dim=768,
        # loss_clip=dict(
        #     type='MSELoss',
        #     reduction='mean',
        # ),
    ),
    train_cfg=dict(
        assigner=dict(
            cls_cost=dict(type='ClassificationCost', weight=0.0),  # no clip cost for hungarian assigner
        )
    ),
)
