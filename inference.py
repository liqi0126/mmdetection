# -*- coding: utf-8 -*-


import os
import numpy as np
import glob

from PIL import Image

from fire import Fire

import cv2

from mmdet.apis import init_detector, inference_detector

import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from imgviz import label_colormap
from mmdet.core import get_classes

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from palette import COCO_STUFF_PALETTE

INSTANCE_OFFSET = 1000

config_file = 'configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_clip_only.py'
checkpoint_file = 'work_dirs/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_clip_only/latest.pth'

import torch
import torch.nn.functional as F

def main(from_folder='/data/Replica_Dataset/room_0/Sequence_1/rgb',
         to_folder='tmp',
         conf=0.3,
         mask_low=True,
         use_conf=False):

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    model.CLASSES = get_classes('coco')

    palette = np.array(COCO_STUFF_PALETTE)

    inst_palette = label_colormap(50)

    os.makedirs(to_folder, exist_ok=True)

    videos = glob.glob(f"{from_folder}/rgb_*.png")
    for i, frame in enumerate(videos):
        idx = int(frame.split('/')[-1][4:-4])
        print(f"process frame {idx}")

        result = inference_detector(model, frame)
        panoptic = result['pan_results']

        sem_seg = panoptic % INSTANCE_OFFSET + 1
        sem_seg[sem_seg == 134] = 0  # void
        ins_seg = panoptic // INSTANCE_OFFSET

        sem_seg = sem_seg.astype('uint8')
        ins_seg = ins_seg.astype('uint8')

        ins_viz = inst_palette[ins_seg]

        sem_viz = np.zeros((*sem_seg.shape, 3), dtype='uint8')

        sem_viz[sem_seg > 0] = palette[sem_seg[sem_seg > 0] - 1]

        cv2.imwrite(f'{to_folder}/semantic_class_{idx}.png', sem_seg)
        cv2.imwrite(f'{to_folder}/semantic_instance_{idx}.png', ins_seg)
        cv2.imwrite(f'{to_folder}/semantic_viz_{idx}.png', sem_viz)
        cv2.imwrite(f'{to_folder}/instance_viz_{idx}.png', ins_viz)
        model.show_result(frame, result, out_file=f'{to_folder}/pano_{idx}.jpg')

if __name__ == '__main__':
    Fire(main)
