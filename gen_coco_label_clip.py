# -*- coding: utf-8 -*-

import json
import numpy as np

from fire import Fire
import clip

import torch

# CLASSES = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
#     ' truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#     'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#     'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#     'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
#     'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
#     'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
#     'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
#     'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
#     'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
#     'wall-wood', 'water-other', 'window-blind', 'window-other',
#     'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
#     'cabinet-merged', 'table-merged', 'floor-other-merged',
#     'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
#     'paper-merged', 'food-other-merged', 'building-other-merged',
#     'rock-merged', 'wall-other-merged', 'rug-merged'
# ] + ['nothing']


CLASSES = [
    'a person', 'a bicycle', 'a car', 'a motorcycle', 'an airplane', 'a bus', 'a train',
    'a truck', 'a boat', 'a traffic light', 'a fire hydrant', 'a stop sign',
    'a parking meter', 'a bench', 'a bird', 'a cat', 'a dog', 'a horse', 'sheep',
    'a cow', 'an elephant', 'a bear', 'a zebra', 'a giraffe', 'a backpack', 'an umbrella',
    'a handbag', 'a tie', 'a suitcase', 'a frisbee', 'a skis', 'a snowboard',
    'a sports ball', 'a kite', 'a baseball bat', 'a baseball glove', 'a skateboard',
    'a surfboard', 'a tennis racket', 'a bottle', 'a wine glass', 'a cup', 'a fork',
    'a knife', 'a spoon', 'a bowl', 'a banana', 'an apple', 'a sandwich', 'an orange',
    'broccoli', 'a carrot', 'a hot dog', 'pizza', 'a donut', 'a cake', 'a chair',
    'a couch', 'a potted plant', 'a bed', 'a dining table', 'a toilet', 'a TV',
    'a laptop', 'a mouse', 'a remote', 'a keyboard', 'a cell phone', 'a microwave',
    'an oven', 'a toaster', 'a sink', 'a refrigerator', 'a book', 'a clock', 'a vase',
    'a pair of scissors', 'a teddy bear', 'a hair drier', 'a toothbrush', 'a banner',
    'a blanket', 'a bridge', 'a cardboard', 'a counter', 'a curtain', 'a door',
    'a wood floor', 'a flower', 'fruit', 'gravel', 'a house', 'a light',
    'a mirror', 'a net', 'a pillow', 'a platform', 'a playingfield',
    'a railroad', 'a river', 'a road', 'a roof', 'sand', 'a sea', 'a shelf', 'snow',
    'stairs', 'a tent', 'a towel', 'a brick wall', 'a stone wall', 'a tile wall',
    'a wood wall', 'water', 'a blind', 'a window',
    'a tree', 'a fence', 'a ceiling', 'sky',
    'a cabinet', 'a table', 'a floor',
    'pavement', 'a mountain', 'grass', 'dirt',
    'paper', 'food', 'a building',
    'a rock', 'a wall', 'rug'
] + ['nothing']


def main(base_dir='/data/coco/annotations'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14@336px', device)

    text = torch.cat([clip.tokenize(c) for c in CLASSES]).cuda()
    with torch.no_grad():
        text_feat = model.encode_text(text).cpu().numpy()

    with open(f'{base_dir}/coco_panoptic_clip.npy', 'wb') as f:
        np.save(f, text_feat)


if __name__ == '__main__':
    Fire(main)
