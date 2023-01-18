# -*- coding: utf-8 -*-

from gen_coco_label_clip import CLASSES
import os
import clip
import json
import torch.nn.functional as F

from PIL import Image

import numpy as np
from copy import deepcopy
import glob

from panopticapi.utils import rgb2id
import mmcv

from PIL import Image

from fire import Fire

import cv2


import torchvision.transforms as T

from mmdet.apis import init_detector, inference_detector
from imgviz import label_colormap

import torch
from mmcv.parallel import collate, scatter


INSTANCE_OFFSET = 1000

config_file = 'configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_clip_only.py'
checkpoint_file = 'work_dirs/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_clip_only/latest.pth'
# checkpoint_file = 'checkpoints/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth'
checkpoint_file = 'work_dirs/old/latest.pth'


def get_logits(img_features, text_features):
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = img_features @ text_features.float().t()

    return logits_per_image


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return T.Compose([
        T.Resize((n_px, n_px), interpolation=T.InterpolationMode.BICUBIC),
        # T.CenterCrop(n_px),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def highlight_mask(img, mask):
    np_img = np.array(img)
    np_img[~mask] = 255
    x, y, w, h = cv2.boundingRect(mask.astype('uint8'))
    np_img = np_img[y:y+h, x:x+w]
    img_highlight = Image.fromarray(np_img)
    return img_highlight


def main(img_folder='/data/coco/unlabeled2017',
         ann_in_file='/data/coco/annotations/image_info_unlabeled2017.json',
         ann_out_file='/data/coco/annotations/panoptic_unlabeled2017.json',
         ann_mask_folder='/data/coco/annotations/panoptic_unlabeled2017',
         # vis_folder='tmp'
         vis_folder='full_mask'
         ):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model, preprocess = clip.load('ViT-L/14@336px', device)
    preprocess = _transform(336)

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    text = torch.cat([clip.tokenize(c) for c in CLASSES[:-1]]).cuda()
    with torch.no_grad():
        text_feat = clip_model.encode_text(text)
    img_preprocess = T.Compose([
        T.Resize(size=336, interpolation=T.InterpolationMode.BICUBIC),
        # T.CenterCrop(size=(224, 224))
    ])

    with open(ann_in_file, 'r') as f:
        ann_json = json.load(f)

    annotations = []

    anno_id = 1
    mask_id = 1
    for i, image_dict in enumerate(ann_json['images']):
        anno_dict = {}

        img_path = f"{img_folder}/{image_dict['file_name']}"

        img = Image.open(img_path)

        result = inference_detector(model, img_path)
        model.show_result(img_path, result, out_file=f'{vis_folder}/pano_{i}.jpg')

        pan = result['pan_results']
        pan_copy = deepcopy(pan)
        idx = 1
        for c in np.unique(pan_copy):
            m = pan_copy == c
            np_img = np.array(img)
            # np_img[~m] = np.random.randint(256, size=np_img[~m].shape)
            # np_img[~m] = 255
            np_img[~m] = 0
            x, y, w, h = cv2.boundingRect(m.astype('uint8'))
            np_img = np_img[y:y+h, x:x+w]
            img_masked = Image.fromarray(np_img)
            img_masked_out = img_preprocess(img_masked)
            img_masked_out.save(f'{vis_folder}/tmp_{idx}.png')
            idx = idx + 1
            img_masked_processed = preprocess(img_masked).unsqueeze(0).cuda()
            with torch.no_grad():
                img_features = clip_model.encode_image(img_masked_processed).float()
            logits = get_logits(img_features, text_feat).squeeze()
            pan[m] = 1000 * (c // 1000) + logits.argmax().item()
        model.show_result(img_path, {'pan_results': pan}, out_file=f'{vis_folder}/clip_{i}.jpg')

    #     mask = result['mask_results'].cpu().numpy()
    #     mask_copy = deepcopy(mask)
    #     mask[:] = 0

    #     segments_info = []
    #     for c in np.unique(mask_copy):
    #         if c == 0:
    #             continue
    #         segment_info = {}
    #         m = mask_copy == c
    #         mask[m] = anno_id
    #         segment_info['id'] = anno_id
    #         segment_info['category_id'] = 0
    #         segment_info['iscrowd'] = 0
    #         x, y, w, h = cv2.boundingRect(m.astype('uint8'))
    #         segment_info['bbox'] = [x, y, w, h]
    #         segment_info['area'] = m.sum()
    #         np_img = np.array(img)
    #         np_img[~m] = 255
    #         np_img = np_img[y:y+h, x:x+w]
    #         img_masked = Image.fromarray(np_img)
    #         img_masked_processed = preprocess(img_masked).unsqueeze(0).cuda()
    #         with torch.no_grad():
    #             img_features = clip_model.encode_image(img_masked_processed)
    #         segment_info['clip_feat'] = img_features.cpu().float().numpy().squeeze()
    #         segments_info.append(segment_info)
    #         anno_id += 1

    #     mask_file_name = f"{mask_id:012}.png"
    #     anno_dict['segments_info'] = segments_info
    #     anno_dict['image_id'] = image_dict['id']
    #     anno_dict['file_name'] = mask_file_name

    #     rgb_mask = np.zeros((*mask.shape, 3), dtype='uint8')

    #     rgb_mask[..., 2] = mask % 256
    #     rgb_mask[..., 1] = (mask // 256) % 256
    #     rgb_mask[..., 0] = (mask // (256 * 256)) % 256

    #     cv2.imwrite(f'{ann_mask_folder}/{mask_file_name}', rgb_mask)

    #     mask_id += 1

    #     annotations.append(anno_dict)

        print(f"process frame {i} / {len(ann_json['images'])} done")

    # ann_json['annotations'] = annotations

    # with open(ann_out_file, 'w') as f:
    #     json.dump(ann_json, f, cls=NpEncoder)


if __name__ == '__main__':
    Fire(main)
