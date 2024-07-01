# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Support synchronized crop and valid_pseudo_mask
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import json
import pickle
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from . import CityscapesDataset
from .builder import DATASETS
from mmcv import Config

def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    original_freq = freq
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy(), original_freq


def get_crop_bbox(img_size, crop_size):
    """Randomly get a crop bounding box."""
    assert len(img_size) == len(crop_size)
    assert len(img_size) == 2
    margin_h = max(img_size[0] - crop_size[0], 0)
    margin_w = max(img_size[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


@DATASETS.register_module()
class UDADataset(object):

    def __init__(self, source, target, cfg):
        self.source = source
        # mmcv.print_log(f'self.source: {self.source}', 'mmseg')
        self.target = target
        # mmcv.print_log(f'self.target: {self.target}', 'mmseg')
        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.ignore_index == source.ignore_index
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

        if cfg.cdmix:
            with open(osp.join(source.data_root, 'cls_prob_distribution_diou.pkl'), 'rb') as file:
                self.cls_dist = pickle.load(file)
            
            self.get_cls_relation(self.cls_dist)

        self.sync_crop_size = cfg.get('sync_crop_size')
        rcs_cfg = cfg.get('rare_class_sampling')
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            # 更改過 self.freq
            self.rcs_classes, self.rcs_classprob, self.freq = get_rcs_class_probs(
                cfg['source']['data_root'], self.rcs_class_temp)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')
            # cfg.setdefault('uda', {})
            # cfg.setdefault('rcs_classes', self.rcs_classes)
            # cfg.setdefault('rcs_classprob', self.rcs_classprob)
            # cfg.setdefault('freq', self.freq)
            # cfg['rcs_classes'] = self.rcs_classes
            # cfg['rcs_classprob'] = self.rcs_classprob
            # cfg['freq'] = self.freq
           

            with open(
                    osp.join(cfg['source']['data_root'],
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def get_cls_relation(self, cls_dist):
        """
            get class relation for each class, lower distance means more related

            return: sorted class relation 
            cls_dist['relation'] = {
                cls_num: [(cls_num, distance), (2, 0.2), ...],
                ...
            }
        """
        cls_realtion = {}
        prob = cls_dist['prob']
        num_cls = len(self.CLASSES)

        for cls_a in range(num_cls):
            cls_realtion[cls_a] = []
            for cls_b in range(num_cls):
                if cls_a != cls_b:
                    p = prob[cls_a, cls_b]
                    indices = np.arange(len(p))

                    cls_realtion[cls_a].append(
                        (cls_b, np.sum(indices * p))
                    )
            
            cls_realtion[cls_a] = sorted(cls_realtion[cls_a], key=lambda x: x[1])
            cls_realtion[cls_a] = torch.tensor(cls_realtion[cls_a])
        
        cls_dist['relation'] = cls_realtion

    def synchronized_crop(self, s1, s2):
        if self.sync_crop_size is None:
            return s1, s2
        orig_crop_size = s1['img'].data.shape[1:]
        crop_y1, crop_y2, crop_x1, crop_x2 = get_crop_bbox(
            orig_crop_size, self.sync_crop_size)
        for i, s in enumerate([s1, s2]):
            for key in ['img', 'gt_semantic_seg', 'valid_pseudo_mask']:
                if key not in s:
                    continue
                s[key] = DC(
                    s[key].data[:, crop_y1:crop_y2, crop_x1:crop_x2],
                    stack=s[key]._stack)
        return s1, s2

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s1 = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]

        # Before synchronized_crop(), s1 and s2 are cropped independently from
        # the entire image when calling s1=self.source[i1] and
        # s2=self.target[i2]. This corresponds to the original implementation
        # in DACS and DAFormer. However, in both papers only large crops were
        # used.
        # In some experiments of the HRDA paper, smaller crop sizes are
        # necessary. We found that independent small crops do not work
        # well with ClassMix (see dacs.py) as the content layout does not
        # match. Therefore, we use synchronized cropping, where the same
        # subcrop region is applied to s1 and s2.
        s1, s2 = self.synchronized_crop(s1, s2)
        # mmcv.print_log(f's2: {s2}', 'mmseg')
        # non_255_mask = s2['gt_semantic_seg'] != 255
        # if not isinstance(non_255_mask, torch.Tensor):
        #     non_255_mask = torch.tensor(non_255_mask, dtype=torch.int)
        # num_non_255_values = torch.sum(non_255_mask).item()
        # mmcv.print_log(f'Number of values not equal to 255: {num_non_255_values}', 'mmseg')
        out = {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img']
        }
        # out = {
        #     **s1, **s2
        # }
        # mmcv.print_log(f'out_origin: {out}', 'mmseg')
        if 'valid_pseudo_mask' in s2:
            out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
        out['target_gt_semantic_seg'] = s2['gt_semantic_seg']
        
        # mmcv.print_log(f'out: {out}', 'mmseg')
        return out

    def __getitem__(self, idx):
        if self.rcs_enabled:
            out = self.get_rare_class_sample()
        else:
            s1 = self.source[idx // len(self.target)]
            s2 = self.target[idx % len(self.target)]
            s1, s2 = self.synchronized_crop(s1, s2)
            out = {
                **s1, 'target_img_metas': s2['img_metas'],
                'target_img': s2['img']
            }
            
            if self.debug:
                out['target_gt'] = s2['gt_semantic_seg']
            
            if 'valid_pseudo_mask' in s2:
                out['valid_pseudo_mask'] = s2['valid_pseudo_mask']

        if hasattr(self, 'cls_dist'):
            out['cls_dist'] = self.cls_dist

        return out

    def __len__(self):
        return len(self.source) * len(self.target)