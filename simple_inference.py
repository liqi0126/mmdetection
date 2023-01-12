# -*- coding: utf-8 -*-

from fire import Fire
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py'
checkpoint_file = 'checkpoints/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth'

def main(frame):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    result = inference_detector(model, frame)
    model.show_result(frame, result, out_file=f'pano.jpg')

if __name__ == '__main__':
    Fire(main)

