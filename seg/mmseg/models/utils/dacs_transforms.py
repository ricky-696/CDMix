# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn

import os
import sys
sys.path.append('.')

from matplotlib import pyplot as plt
from torch.distributions import Categorical
from tools.classes_distance.utils import diou_loss, mask_to_bbox
from mmseg.models.utils.visualization import subplotimg


def extract_window_v2(all_windows, cls_diou_losses, topk_cls_dist, topk_cls_dist_indices):
    """
    N: n_windows
    C: n_classes
    K: top_k

    Shapes:
    * all_windows: [N, 4]
    * cls_diou_losses: [C, N]
    * topk_cls_dist: [K, 2]
    * topk_cls_dist_indices: [K,]
    """
    cls_diou_losses = cls_diou_losses[topk_cls_dist_indices].T  # [N, K], where the axis of K is ordered.
    min_cls_diou_losses, max_cls_diou_losses = topk_cls_dist.T  # [K,], [K,]
    k_conditions = (min_cls_diou_losses <= cls_diou_losses) & (cls_diou_losses <= max_cls_diou_losses)  # [N, K]
    matched_k = torch.all(k_conditions, dim=1)  # [N,]
    matched_windows = all_windows[matched_k]

    if len(matched_windows) > 0:
        return matched_windows[0]
    else:
        return None


def vis_mixing_cls(source_gt, mixing_gt, ori_target_gt, mixed_target_gt):
    rows, cols = 1, 4
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows),
        gridspec_kw={
            'hspace': 0.1,
            'wspace': 0,
            'top': 0.9,
            'bottom': 0.1,
            'right': 1,
            'left': 0
        },
    )
    subplotimg(
        axs[0],
        source_gt,
        'Source Seg GT',
        cmap='cityscapes')
    subplotimg(
        axs[1],
        mixing_gt,
        'mixing_cls',
        cmap='cityscapes')
    subplotimg(
        axs[2],
        ori_target_gt,
        'ori_target_gt',
        cmap='cityscapes')
    subplotimg(
        axs[3],
        mixed_target_gt,
        'mixed_target_gt',
        cmap='cityscapes')
    
    plt.savefig('source_gt.jpg')
    plt.close()


def seg_sliding_windows(source_cls, cls_mask, gt_mask, cls_dist_mat, cls_relation, step_size=1) -> torch.tensor:
    """
        Args:
            source_cls: int, source class
            cls_mask: tensor, shape (H, W), binary mask of source class
            gt_mask: tensor, shape (H, W), groundtruth mask
            cls_dist_mat: dict, key is tuple of (source_cls, cls), value is tuple of (min_val, max_val)

        return:
            window: tensor, shape (4,), x1, y1, x2, y2
    """

    mask_bbox = mask_to_bbox(cls_mask)[0] # x1, y1, x2, y2

    # Compute the window size
    window_size = (mask_bbox[2] - mask_bbox[0], mask_bbox[3] - mask_bbox[1])
    # Compute the coordinates of the sliding windows
    x_coords = torch.arange(0, gt_mask.shape[1] - window_size[0] + 1, step_size)
    y_coords = torch.arange(0, gt_mask.shape[0] - window_size[1] + 1, step_size)
    # Create grid of coordinates
    xx, yy = torch.meshgrid(x_coords, y_coords)
    # Flatten coordinates
    flat_x = xx.flatten()
    flat_y = yy.flatten()

    # shape: [num_windows, 4]
    all_windows = torch.stack((flat_x, flat_y, flat_x + window_size[0], flat_y + window_size[1])).t()
    
    # get groundtruth cls, shape: [num_cls]
    gt_cls = torch.unique(gt_mask)
    gt_cls = gt_cls[gt_cls != 255].cpu().numpy() # remove background cls

    cls_diou_losses = {}
    for cls in gt_cls:
        gt_bbox = mask_to_bbox(gt_mask==cls).expand(all_windows.shape[0], -1)

        # diou_losses: the diou loss for all sliding windows, shape: [num_windows]
        diou_losses, ious = diou_loss(gt_bbox, all_windows)
        cls_diou_losses[cls] = diou_losses
        
    # Choose top k cls relation
    top_k = 2
    topk_cls_relation = [cls[0] for cls in cls_relation[source_cls] if cls[0] in cls_diou_losses][:top_k]
    topk_cls_dist = [(cls, cls_dist_mat[(source_cls, cls)]) for cls in topk_cls_relation]

    # check if exist one windows axis all in range of topk_cls_dist, return axis
    diou_checks = torch.ones(all_windows.shape[0], dtype=torch.bool)
    for cls, (min_val, max_val) in topk_cls_dist:
        diou_losses = cls_diou_losses[cls]
        diou_checks &= (diou_losses >= min_val) & (diou_losses <= max_val)

    valid_windows = all_windows[diou_checks]

    if valid_windows.shape[0] > 0:
        # print(
        #     'source_cls: ', cls_name[source_cls], 
        #     'topk_cls_dist-1: ', cls_name[topk_cls_dist[0][0]], topk_cls_dist[0][1],
        #     'topk_cls_dist-2: ', cls_name[topk_cls_dist[1][0]], topk_cls_dist[1][1],
        # )
        
        sample_idx = torch.randint(0, valid_windows.shape[0], (1,)).item()
        return valid_windows[sample_idx], mask_bbox
    else:
        return None, None


def cls_dist_mix(mask, ignore_cls, data=None, target=None, cls_dist=None):
    if mask is None:
        return data, target
    
    source_img, target_img = data
    source_gt, target_gt = target
    C, H, W = source_img.shape

    gt_mask = mask.view(source_gt.shape)
    img_mask = gt_mask.unsqueeze(0).expand(C, -1, -1)

    # ignore some big class
    mixing_gt = source_gt * gt_mask
    mask = (mixing_gt[..., None] == ignore_cls).any(-1)
    mixing_gt[mask] = 255

    # get cls prob
    cls_dist_mat = {}
    for keys, value in cls_dist['prob'].items():
        if keys[0] == keys[1]:
            cls_dist_mat[keys] = torch.tensor(0)
        else:
            distribution = Categorical(probs=torch.tensor(value))
            bin_idx = min(distribution.sample((3, )))

            # 0-indexed, 0: [0, 0.1), 1: [0.1, 0.2), ..., 20: [1.9, 2.0]
            cls_dist_mat[keys] = (cls_dist['bin_edges'][bin_idx], cls_dist['bin_edges'][bin_idx + 1])

    # for every cls, chossen the best mixing axis
    mixed = False
    for cls in torch.unique(mixing_gt):
        if cls != 255:
            cls_mask = mixing_gt==cls
            mix_axis, ori_axis = seg_sliding_windows(
                source_cls=cls.item(), 
                cls_mask=cls_mask,
                gt_mask=target_gt,
                cls_dist_mat=cls_dist_mat,
                cls_relation=cls_dist['relation'],
            )

            if mix_axis is not None:
                mixed = True
                ori_x1, ori_y1, ori_x2, ori_y2 = ori_axis
                mix_x1, mix_y1, mix_x2, mix_y2 = mix_axis

                masked_img = source_img * cls_mask
                mix_img = masked_img[:, ori_y1:ori_y2, ori_x1:ori_x2]
                target_img[:, mix_y1:mix_y2, mix_x1:mix_x2].copy_(
                    torch.where(mix_img != 0, mix_img, target_img[:, mix_y1:mix_y2, mix_x1:mix_x2])
                )

                ori_target_gt = target_gt.clone()
                
                masked_gt = source_gt * cls_mask
                mix_gt = masked_gt[ori_y1:ori_y2, ori_x1:ori_x2]
                target_gt[mix_y1:mix_y2, mix_x1:mix_x2].copy_(
                    torch.where(mix_gt != 0, mix_gt, target_gt[mix_y1:mix_y2, mix_x1:mix_x2])
                )
                
                vis_mix_cls = masked_gt.clone()
                vis_mix_cls[vis_mix_cls == 0] = 255
                vis_mixing_cls(source_gt, vis_mix_cls, ori_target_gt, target_gt)
                
                del masked_img, mix_img, masked_gt, mix_gt, ori_target_gt, vis_mix_cls
        
    del source_img, target_img, source_gt, target_gt, gt_mask, img_mask, cls_dist_mat
    torch.cuda.empty_cache()
        
    # 一樣的資料如果使用one_mix這個func回傳data, target進去模型forward就不會OOM，我猜是因為one_mix都是inplace的操作，
    # 所以在forward時並不會新增新的向量，目前猜測是模型forward時會把masked_img, mix_img丟進去GPU，所以OOM
    return data, target, mixed


def strong_transform(param, data=None, target=None, cls_dist=None):
    """
        Args:
            param: dict, parameters for strong transform
            data: tensor, images, shape (2, C, H, W), where 2 is source and target
            target: tensor, groundtruths, shape (2, H, W), where 2 is source and target

        Returns:
            data: tensor, images, shape (2, C, H, W), where 2 is source and target
            target: tensor, groundtruths, shape (2, H, W), where 2 is source and target
    """
    assert ((data is not None) or (target is not None))

    if cls_dist is not None:
        data, target, mixed = cls_dist_mix(
            mask=param['mix'], 
            ignore_cls=param['ignore_cls'], 
            data=data, target=target, cls_dist=cls_dist,
        )
        if not mixed:
            data, target = one_mix(mask=param['mix'], data=data, target=target)
    else:
        data, target = one_mix(mask=param['mix'], data=data, target=target)

    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    
    import random
    from mmcv.utils import Config
    from mmseg.datasets import build_dataset
    
    cls_name = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle',
    ]

    cfg = Config.fromfile(
        'configs/_base_/datasets/uda_cityscapes_to_acdc_512x512_debug.py'
    )
    
    dataset = build_dataset(cfg.data.train)
    
    for data in dataset:
        means, stds = get_mean_std([data['img_metas'].data], data['img'].data.device)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': cfg['color_jitter_strength'][0],
            'color_jitter_p': cfg['color_jitter_probability'][0],
            'blur': random.uniform(0, 1) if cfg['blur'] else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        
        mix_masks = get_class_masks(data['gt_semantic_seg'].data)
        
        strong_parameters['mix'] = mix_masks[0]
        strong_parameters['ignore_cls'] = torch.tensor([0, 1, 10], dtype=strong_parameters['mix'].dtype) # ignore road, sidewalk, sky
        
        mixed_img, mixed_lbl = strong_transform(
            strong_parameters,
            data=torch.stack((data['img'].data, data['target_img'].data)),
            target=torch.stack(
                (data['gt_semantic_seg'].data[0], data['target_gt'].data[0])),
            cls_dist=data['cls_dist'],
        )