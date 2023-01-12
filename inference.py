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

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from palette import COCO_STUFF_PALETTE

INSTANCE_OFFSET = 1000

config_file = 'configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py'
checkpoint_file = 'checkpoints/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth'

import torch
import torch.nn.functional as F

def main(from_folder='/data/Replica_Dataset/room_0/Sequence_1/rgb',
         to_folder='/data/Replica_Dataset/room_0/Sequence_1/mask2former',
         conf=0.3,
         mask_low=True,
         use_conf=False):

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    palette = np.array(COCO_STUFF_PALETTE)

    logits_2_uncertainty = lambda x: torch.sum(-F.log_softmax(x, dim=0)*F.softmax(x, dim=0), dim=0, keepdim=False)

    inst_palette = label_colormap(50)

    os.makedirs(to_folder, exist_ok=True)

    videos = glob.glob(f"{from_folder}/rgb_*.png")
    for i, frame in enumerate(videos):
        idx = int(frame.split('/')[-1][4:-4])
        print(f"process frame {idx}")

        result = inference_detector(model, frame)
        panoptic = result['pan_results']
        prob_masks = result['prob_masks']

        if prob_masks.shape[0] == 0:
            confidence = np.zeros(panoptic.shape)
        else:
            confidence = prob_masks.max(0)

        if use_conf:
            mask = confidence > conf
        else:
            entropy = logits_2_uncertainty(torch.tensor(prob_masks))
            mask = entropy < entropy.mean()

        sem_seg = panoptic % INSTANCE_OFFSET + 1
        sem_seg[sem_seg == 134] = 0  # void
        if mask_low:
            sem_seg[~mask] = 0
        ins_seg = panoptic // INSTANCE_OFFSET
        if mask_low:
            ins_seg[~mask] = 0

        sem_seg = sem_seg.astype('uint8')
        ins_seg = ins_seg.astype('uint8')

        ins_viz = inst_palette[ins_seg]

        sem_viz = np.zeros((*sem_seg.shape, 3), dtype='uint8')

        sem_viz[sem_seg > 0] = palette[sem_seg[sem_seg > 0] - 1]

        np.savez_compressed(f'{to_folder}/confidence_{idx}.npz', confidence=confidence)
        cv2.imwrite(f'{to_folder}/semantic_class_{idx}.png', sem_seg)
        cv2.imwrite(f'{to_folder}/semantic_instance_{idx}.png', ins_seg)
        cv2.imwrite(f'{to_folder}/semantic_viz_{idx}.png', sem_viz)
        cv2.imwrite(f'{to_folder}/instance_viz_{idx}.png', ins_viz)
        model.show_result(frame, result, out_file=f'{to_folder}/pano_{idx}.jpg')

if __name__ == '__main__':
    Fire(main)
