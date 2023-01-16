# -*- coding: utf-8 -*-

import os
import numpy as np
from copy import deepcopy
import glob

from PIL import Image

from fire import Fire

import cv2

from mmdet.apis import init_detector, inference_detector
from imgviz import label_colormap

import torch
from mmcv.parallel import collate, scatter


INSTANCE_OFFSET = 1000

config_file = 'configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_clip_only.py'
checkpoint_file = 'work_dirs/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_clip_only/latest.pth'
# checkpoint_file = 'checkpoints/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth'

def main(from_folder='/data/Replica_Dataset/room_0/Sequence_1/rgb',
         to_folder='tmp',
         conf=0.3,
         mask_low=True,
         use_conf=False):

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # kernel = np.ones((7, 7))

    palette = label_colormap(1000)

    os.makedirs(to_folder, exist_ok=True)

    videos = glob.glob(f"{from_folder}/rgb_*.png")
    for i, frame in enumerate(videos):

        img = cv2.imread(frame)

        idx = int(frame.split('/')[-1][4:-4])

        result = inference_detector(model, frame)
        mask = result['mask_results'].cpu().numpy()

        for c in np.unique(mask):
            img_copy = deepcopy(img)
            m = mask == c
            # m = cv2.filter2D(m.astype(float), ddepth=-1, kernel=kernel) > 0

            img_copy[~m] = 0
            cv2.imwrite(f'{to_folder}/mask_{idx}_class_{c}.png', img_copy)

        cv2.imwrite(f'{to_folder}/mask_{idx}.png', palette[mask])

        # for i in range():
        #     cv2.imwrite(f'{to_folder}/semantic_class_{idx}.png', sem_seg)

        model.show_result(frame, result, out_file=f'{to_folder}/pano_{idx}.jpg')

        print(f"process frame {idx} done")

if __name__ == '__main__':
    Fire(main)

