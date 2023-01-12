# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import cv2

import torch
import torch.nn.functional as F

import networkx as nx

from copy import deepcopy

from PIL import Image

from fire import Fire

import torchvision.transforms as T


from mmdet.apis import init_detector, inference_detector
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from imgviz import label_colormap

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from palette import COCO_STUFF_PALETTE

INSTANCE_OFFSET = 1000

config_file = 'configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py'
checkpoint_file = 'checkpoints/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth'


color_jitter = T.ColorJitter(brightness=.5, contrast=.3, saturation=.2, hue=.2)
horizontal_flip = T.RandomHorizontalFlip(p=1)
gaussian_blur = T.GaussianBlur(kernel_size=(51, 91), sigma=2)

def augmentation(img):
    flip = False
    if np.random.random() < .5:
        img = color_jitter(img)
    if np.random.random() < .5:
        img = horizontal_flip(img)
        flip = True
    if np.random.random() < .3:
        img = gaussian_blur(img)

    return img, flip

def get_soft_iou(mask_pred):
    n = mask_pred.shape[0]
    soft_iou = torch.zeros(n, n)

    for i in range(n):
        for j in range(n):
            min_sum = torch.min(mask_pred[i], mask_pred[j]).sum()
            max_sum = torch.max(mask_pred[i], mask_pred[j]).sum()
            soft_iou[i, j] = min_sum / max_sum

    return soft_iou

def connect_masks(mask_cls, mask_pred):
    soft_iou = get_soft_iou(mask_pred)

    G = nx.Graph()
    G.add_nodes_from(range(soft_iou.shape[0]))
    G.add_edges_from(np.stack(np.where(soft_iou > .5)).T)

    connected_cls = []
    connected_masks = []
    for i, c in enumerate(nx.connected_components(G)):
        sg = G.subgraph(c)
        nodes = list(sg.nodes())

        cls = mask_cls[list(nodes)].mean(0)
        mask = mask_pred[list(nodes)].mean(0)

        connected_cls.append(cls)
        connected_masks.append(mask)

    connected_cls = np.stack(connected_cls, 0)
    connected_masks = np.stack(connected_masks, 0)

    return connected_cls, connected_masks


def panoptic_post_process(mask_cls, mask_pred, object_mask_thr=.8, iou_thr=.8, filter_low_score=True, num_classes=133, num_things_classes=80):
    mask_cls = F.softmax(mask_cls)
    scores, labels = mask_cls.max(-1)
    mask_pred = mask_pred.sigmoid()

    keep = labels.ne(num_classes) & (scores > object_mask_thr)
    mask_cls = mask_cls[keep]
    mask_pred = mask_pred[keep]

    mask_cls, mask_pred = connect_masks(mask_cls, mask_pred)
    mask_cls = torch.tensor(mask_cls)
    mask_pred = torch.tensor(mask_pred)

    scores, labels = mask_cls.max(-1)

    keep = labels.ne(num_classes) & (scores > object_mask_thr)
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]

    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.full((h, w),
                              num_classes,
                              dtype=torch.int32,
                              device=cur_masks.device)

    if cur_masks.shape[0] == 0:
        # We didn't detect any mask :(
        pass
    else:
        cur_mask_ids = cur_prob_masks.argmax(0)
        instance_id = 1
        for k in range(cur_classes.shape[0]):
            pred_class = int(cur_classes[k].item())
            isthing = pred_class < num_things_classes
            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()

            if filter_low_score:
                mask = mask & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < iou_thr:
                    continue

                if not isthing:
                    # different stuff regions of same class will be
                    # merged here, and stuff share the instance_id 0.
                    panoptic_seg[mask] = pred_class
                else:
                    panoptic_seg[mask] = (
                        pred_class + instance_id * INSTANCE_OFFSET)
                    instance_id += 1

    return panoptic_seg, cur_prob_masks


def main(from_folder='/data/Replica_Dataset/room_0/Sequence_1/rgb',
         to_folder='/data/Replica_Dataset/room_0/Sequence_1/mask2former',
         conf=0.3,
         mask_low=False,
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

        mask_cls_group = []
        mask_pred_group = []
        frame = Image.open(frame)

        for j in range(16):
            frame_copy = deepcopy(frame)
            flip = False
            if j > 0:
                frame_copy, flip = augmentation(frame_copy)
            result = inference_detector(model, np.array(frame_copy))
            mask_cls = result["mask_cls_result"]
            mask_pred = result["mask_pred_result"]
            if flip:
                mask_pred = np.flip(mask_pred, -1)
                result['mask_pred_result'] = mask_pred
                result['prob_masks'] = np.flip(result['prob_masks'], -1)
                result['pan_results'] = np.flip(result['pan_results'], -1)
            mask_cls_group.append(mask_cls)
            mask_pred_group.append(mask_pred)

        mask_cls = torch.tensor(np.concatenate(mask_cls_group))
        mask_pred = torch.tensor(np.concatenate(mask_pred_group))

        panoptic, prob_masks = panoptic_post_process(mask_cls, mask_pred)
        panoptic = panoptic.numpy()
        prob_masks = prob_masks.numpy()

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
        model.show_result(np.array(frame), result, out_file=f'{to_folder}/pano_{idx}.jpg')

if __name__ == '__main__':
    Fire(main)
