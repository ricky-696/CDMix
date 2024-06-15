# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation
import sys
sys.path.append('.')

import pickle
import argparse
import numpy as np

from tqdm import tqdm
from mmcv.utils import Config
from mmseg.datasets import build_dataset

from utils import diou_loss, mask_to_bbox


def cnt_prob_distribution(args, cls_dists):
    cls_prob_distribution = {}

    if args.dist_func == 'diou':
        bin_edges = np.arange(0, 2 + args.bin_interval, args.bin_interval)
        bin_edges = np.round(bin_edges, decimals=3)

    for key, value in cls_dists.items():
        if len(value) > 0:
            cls_dist = np.array(value)
            cls_dist_bins = np.digitize(cls_dist, bin_edges, right=True) # 0-indexed, 0: [0, 0.1), 1: [0.1, 0.2), ..., 20: [1.9, 2.0]
            cls_dist_counts = np.bincount(cls_dist_bins, minlength=len(bin_edges) - 1)

            if len(cls_dist) <= 0:
                print('here')

            cls_dist_distribution = cls_dist_counts / len(cls_dist)

            cls_prob_distribution[key] = cls_dist_distribution
        else:
            cls_prob_distribution[key] = np.zeros(len(bin_edges) - 1) # prob is 0

    res = {
        'prob': cls_prob_distribution,
        'bin_edges': bin_edges
    }

    return res


def update_cls_dists(args, cls_dists, gt):
    if args.dist_func == 'diou':
        dist_func = diou_loss

    for i in range(args.num_classes):
        for j in range(i + 1, args.num_classes):
            i_mask = gt == i
            j_mask = gt == j
            
            if i_mask.sum() != 0 and j_mask.sum() != 0:
                loss, iou = dist_func(
                    mask_to_bbox(i_mask), 
                    mask_to_bbox(j_mask)
                )
                
                cls_dists[i, j].append(loss.item())
                cls_dists[j, i].append(loss.item())
        
        return


def count_classes_distance(args, dataset):
    """
        Count classes_distance in the dataset

        Args:
            args: arguments
            dataset: dataset to count classes_distance

        Return:
            cls_dists: shape(num_classes, num_classes, num_imgs)
    """

    args.num_classes = len(dataset.CLASSES)
    cls_dists = {}

    for i in range(args.num_classes):
        for j in range(args.num_classes):
            cls_dists[i, j] = []
            cls_dists[j, i] = []

    for data in tqdm(dataset, desc='Counting classes distance'):
        update_cls_dists(args, cls_dists, data['gt_semantic_seg'].data)
        

    return cls_dists


def parse_args():
    parser = argparse.ArgumentParser(
        description='Count classes distance')
    parser.add_argument('--config', type=str, default='configs/_base_/datasets/uda_cityscapes_to_acdc_512x512.py', help='dataset config')
    parser.add_argument('--dist_func', type=str, default='diou', help='dist_func')
    parser.add_argument('--bin_interval', type=float, default=0.1, help='bin_interval')
    parser.add_argument('--dataset', type=str, nargs='+', default=['source', 'target'], help='dataset type: source, target, or both')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    for data_type in args.dataset:
        if data_type not in ['source', 'target']:
            raise ValueError('Dataset type must be either source or target')
        
        data = getattr(cfg.data.train, data_type)

        dataset = build_dataset(data)
        cls_dists = count_classes_distance(args, dataset)

        # Maybe ToDo: count each cls_dists's mean and std
        cls_prob_distribution = cnt_prob_distribution(args, cls_dists)

        with open(f'{data.data_root}/cls_dists_{args.dist_func}.pkl', 'wb') as f:
            pickle.dump(cls_dists, f)

        with open(f'{data.data_root}/cls_prob_distribution_{args.dist_func}.pkl', 'wb') as f:
            pickle.dump(cls_prob_distribution, f)


if __name__ == '__main__':
    # debug
    # args = parse_args()
    # cls_dists = {}
    # for i in range(10):
    #     for j in range(10):
    #         cls_dists[i, j] = [0, 0.5, 1]
    #         cls_dists[j, i] = [0, 0.5, 1]

    # cnt_prob_distribution(args, cls_dists)

    main()
