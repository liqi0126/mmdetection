# -*- coding: utf-8 -*-

import glob
import os

import cv2
from fire import Fire
import numpy as np
from imgviz import label_colormap

from palette import COCO_STUFF_PALETTE

def main(from_folder='/data/Replica_Dataset/room_0/Sequence_1/test_aug',
         to_folder='/data/Replica_Dataset/room_0/Sequence_1/test_aug_conf03',
         conf=0.3):
    videos = glob.glob(f"{from_folder}/semantic_class_*.png")

    palette = np.array(COCO_STUFF_PALETTE)

    inst_palette = label_colormap(50)

    os.makedirs(to_folder, exist_ok=True)

    for i, frame in enumerate(videos):
        idx = int(frame.split('_')[-1][:-4])

        print(f"process frame {idx}")

        confidence = np.load(f'{from_folder}/confidence_{idx}.npz')['confidence']
        sem_seg = cv2.imread(f'{from_folder}/semantic_class_{idx}.png', cv2.IMREAD_UNCHANGED)
        ins_seg = cv2.imread(f'{from_folder}/semantic_instance_{idx}.png', cv2.IMREAD_UNCHANGED)
        sem_viz = cv2.imread(f'{from_folder}/semantic_viz_{idx}.png', cv2.IMREAD_UNCHANGED)
        ins_viz = cv2.imread(f'{from_folder}/semantic_viz_{idx}.png', cv2.IMREAD_UNCHANGED)

        mask = confidence > conf

        sem_seg[~mask] = 0
        ins_seg[~mask] = 0
        sem_viz[~mask] = 0
        ins_viz[~mask] = 0

        np.savez_compressed(f'{to_folder}/confidence_{idx}.npz', confidence=confidence)
        cv2.imwrite(f'{to_folder}/semantic_class_{idx}.png', sem_seg)
        cv2.imwrite(f'{to_folder}/semantic_instance_{idx}.png', ins_seg)
        cv2.imwrite(f'{to_folder}/semantic_viz_{idx}.png', sem_viz)
        cv2.imwrite(f'{to_folder}/instance_viz_{idx}.png', ins_viz)

if __name__ == '__main__':
    Fire(main)
