# -*- coding: utf-8 -*-

import json
from fire import Fire
import clip

import torch


def main(base_dir='/data/coco',
         json_type='val'):

    ann_in_file = f"{base_dir}/annotations/panoptic_{json_type}2017.json"
    ann_out_file = f"{base_dir}/annotations/panoptic_clip_{json_type}2017.json"

    with open(ann_in_file, 'r') as f:
        ann_json = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    class_map = {}

    for cate in ann_json['categories']:
        text = clip.tokenize([cate['name']]).cuda()
        with torch.no_grad():
            text_feat = model.encode_text(text)
        text_feat = text_feat.cpu().numpy().squeeze()
        class_map[cate['id']] = text_feat.tolist()

    for i, anno in enumerate(ann_json['annotations']):
        print(f"{i} / {len(ann_json['annotations'])}")

        for segment in anno['segments_info']:
            segment['clip_feat'] = class_map[segment['category_id']]

    with open(ann_out_file, 'w') as f:
        json.dump(ann_json, f)


if __name__ == '__main__':
    Fire(main)

