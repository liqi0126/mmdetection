# -*- coding: utf-8 -*-

import os
import json
import clip
import numpy as np
from panopticapi.utils import rgb2id
from PIL import Image
from fire import Fire
import torch

import torchvision.transforms as T

from gen_coco_label_clip import CLASSES

from inference import highlight_mask, _transform, _convert_image_to_rgb, get_logits


def main(prefix='train2017'):

    img_folder = f'/data/coco/{prefix}'
    ann_file = f'/data/coco/annotations/panoptic_{prefix}.json'
    ann_mask_folder = f'/data/coco/annotations/panoptic_{prefix}'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model, preprocess = clip.load('ViT-L/14@336px', device)
    preprocess = _transform(336)

    text = torch.cat([clip.tokenize(c) for c in CLASSES[:-1]]).cuda()
    with torch.no_grad():
        text_feat = clip_model.encode_text(text)

    with open(ann_file, 'r') as f:
        ann_json = json.load(f)

    img_dict = {}
    for img in ann_json['images']:
        img_dict[img['id']] = img

    cate_dict = {}
    for i, cate in enumerate(ann_json['categories']):
        cate_dict[cate["id"]] = i

    correct = 0
    total = 0
    total_len = len(ann_json['annotations'])
    for i, ann in enumerate(ann_json['annotations']):
        img = Image.open(os.path.join(img_folder, img_dict[ann['image_id']]['file_name']))
        ann_mask = rgb2id(np.array(Image.open(os.path.join(ann_mask_folder, ann['file_name']))))

        for segment in ann['segments_info']:
            cate = cate_dict[segment['category_id']]
            mask = ann_mask == segment['id']

            assert segment['area'] == mask.sum()

            img_highlight = highlight_mask(img, mask)
            img_processed = preprocess(img_highlight).unsqueeze(0).cuda()

            with torch.no_grad():
                img_features = clip_model.encode_image(img_processed).float()
            logits = get_logits(img_features, text_feat).squeeze()

            correct += cate == logits.argmax().item()
            total += 1

        print(f"{i}\t / {total_len}:\t acc = {correct / total:.5f}")

    print(f"acc = {correct / total}")


if __name__ == '__main__':
    Fire(main)

