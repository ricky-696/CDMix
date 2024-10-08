# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('.')

from matplotlib import pyplot as plt
from torch.distributions import Categorical
from tools.classes_distance.utils import diou_loss, mask_to_bbox
from mmseg.models.utils.visualization import subplotimg, visualize_segmentation_mask

#                       佛祖保佑，Bug全修，準時畢業
# //                            _ooOoo_
# //                           o8888888o
# //                           88" . "88
# //                           (| -_- |)
# //                            O\ = /O
# //                        ____/`---'\____
# //                      .   ' \\| |// `.
# //                       / \\||| : |||// \
# //                     / _||||| -:- |||||- \
# //                       | | \\\ - /// | |
# //                     | \_| ''\---/'' | |
# //                      \ .-\__ `-` ___/-. /
# //                   ___`. .' /--.--\ `. . __
# //                ."" '< `.___\_<|>_/___.' >'"".
# //               | | : `- \`.;`\ _ /`;.`/ - ` : | |
# //                 \ \ `-. \_ __\ /__ _/ .-` / /
# //         ======`-.____`-.___\_____/___.-`____.-'======
# //                            `=---='


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


def vis_mixing_cls(source_img, mixing_img, ori_target_img, mixed_target_img, source_gt, mixing_gt, ori_target_gt, mixed_target_gt):
    rows, cols = 2, 4
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 4 * rows),
        gridspec_kw={
            'hspace': 0.0,
            'wspace': 0.1,
            'top': 0.9,
            'bottom': 0.1,
            'right': 1,
            'left': 0
        },
    )
    subplotimg(
        axs[0][0],
        source_img,
        'Source Image',)
    subplotimg(
        axs[0][1],
        mixing_img,
        'Mixing Class',)
    subplotimg(
        axs[0][2],
        ori_target_img,
        'Original Target Image',)
    subplotimg(
        axs[0][3],
        mixed_target_img,
        'Mixed Target Image',)
    
    subplotimg(
        axs[1][0],
        source_gt,
        'Source Label',
        cmap='cityscapes')
    subplotimg(
        axs[1][1],
        mixing_gt,
        'Mixing Label',
        cmap='cityscapes')
    subplotimg(
        axs[1][2],
        ori_target_gt,
        'Original Target Label',
        cmap='cityscapes')
    subplotimg(
        axs[1][3],
        mixed_target_gt,
        'Mixed Target Label',
        cmap='cityscapes')
    
    plt.savefig('vis.jpg')
    plt.close()


def sliding_window(topk_cls, gt_mask, all_windows, dist=None):
    """
        Args:
            topk_cls: list, top k classes
            gt_mask: tensor, shape (H, W), groundtruth mask
            all_windows: tensor, shape (num_windows, 4), all sliding windows
            dist: dict, key is cls, value is tuple of (min_val, max_val), class distance
            
        return:
            valid_windows: tensor, shape (num_windows, 4), valid sliding windows
    """
    cls_diou_losses = {}
    for cls in topk_cls:
        gt_bbox = mask_to_bbox(gt_mask==cls).expand(all_windows.shape[0], -1)

        # diou_losses: the diou loss for all sliding windows, shape: [num_windows]
        diou_losses, ious = diou_loss(gt_bbox, all_windows)
        cls_diou_losses[cls] = diou_losses

    # check if exist one windows axis all in range of local_dist, return axis
    diou_checks = torch.ones(all_windows.shape[0], dtype=torch.bool) # default is true
    for cls in topk_cls:
        min_val, max_val = dist[cls]
        diou_losses = cls_diou_losses[cls]
        diou_checks &= (diou_losses >= min_val) & (diou_losses <= max_val)

    valid_windows = all_windows[diou_checks]
    
    return valid_windows


def seg_sliding_windows(param, source_cls, cls_mask, gt_mask, local_dist, cls_dist_mat, cls_relation, step_size=1) -> torch.tensor:
    """
        Args:
            param: hyperparameter, include topk, dist_mode
            source_cls: int, source class
            cls_mask: tensor, shape (H, W), binary mask of source class
            gt_mask: tensor, shape (H, W), groundtruth mask
            local_dist: dict, key is source_cls, value is tuple of (min_val, max_val), local distance from source img
            cls_dist_mat: dict, key is tuple of (source_cls, cls), value is tuple of (min_val, max_val)
            cls_relation: dict, key is source_cls, value is list of tuple of (cls, distance)
        return:
            window: tensor, shape (4,), x1, y1, x2, y2
    """
    global_ = False
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
    gt_cls = torch.unique(gt_mask).cpu()
    
    valid_windows = []
    # using local distance to filter out the windows
    if 'local' in param['dist_mode']:
        sorted_local_dist = sorted(local_dist.items(), key=lambda item: item[1][0])
        local_topk_cls = [cls for cls, _ in sorted_local_dist][:param['topk']]
        
        local_topk_set = set(local_topk_cls)
        gt_set = set(gt_cls.tolist())

        if len(local_topk_cls) and local_topk_set.issubset(gt_set):
            valid_windows = sliding_window(local_topk_cls, gt_mask, all_windows, local_dist)
    
    # using global distance from top k cls relation to filter out the windows if local_valid_windows is empty
    if 'global' in param['dist_mode']:
        if len(valid_windows) == 0 or not 'local' in param['dist_mode']:
            cls_relation = cls_relation[source_cls][:, 0].to(gt_mask.dtype)
            topk_gt_cls = [cls.item() for cls in cls_relation if cls in gt_cls][:param['topk']]
            topk_cls_dist = {cls: cls_dist_mat[(source_cls, cls)] for cls in topk_gt_cls}
            
            valid_windows = sliding_window(topk_gt_cls, gt_mask, all_windows, topk_cls_dist)
            global_ = True

    if len(valid_windows) > 0:
        sample_idx = torch.randint(0, valid_windows.shape[0], (1,)).item()
        return valid_windows[sample_idx], mask_bbox, global_
    else:
        return None, None, None


def cls_dist_mix(param, data=None, target=None, weight=None, cls_dist=None):
    mask = param['mix']
    
    if mask is None:
        return data, target, False
    
    source_img, target_img = data
    source_gt, target_gt = target
    source_weight, target_weight = weight
    C, H, W = source_img.shape

    gt_mask = mask.view(source_gt.shape)
    mixing_gt = torch.where(gt_mask == 0, 255, source_gt)

    source_gt_cls = torch.unique(mixing_gt)
    source_gt_cls = source_gt_cls[source_gt_cls != 255]

    # get cls prob
    cls_dist_mat = {}
    if 'global' in param['dist_mode']:
        for keys, value in cls_dist['prob'].items():
            value = torch.from_numpy(value) if type(value) == np.ndarray else value

            if keys[0] == keys[1] or torch.all(torch.unique(value) == 0):
                cls_dist_mat[keys] = torch.tensor(0)
            else:
                distribution = Categorical(probs=value)
                bin_idx = min(distribution.sample((3, )))
                bin_idx = min(bin_idx, len(cls_dist['bin_edges']) - 2)

                # 0-indexed, 0: [0, 0.1), 1: [0.1, 0.2), ..., 20: [1.9, 2.0)
                cls_dist_mat[keys] = (cls_dist['bin_edges'][bin_idx], cls_dist['bin_edges'][bin_idx + 1])

    # for every cls, chossen the best mixing axis
    for cls in source_gt_cls:
        cls_mask = mixing_gt==cls
        
        # count local distance from source img, local_dist[cls] =  (min_val, max_val)
        local_dist = {}
        if 'local' in param['dist_mode']:
            for s_cls in source_gt_cls:
                if cls != s_cls:
                    s_cls = s_cls.item()
                    cls_bbox = mask_to_bbox(cls_mask)
                    s_cls_bbox = mask_to_bbox(mixing_gt==s_cls)

                    diou_losses, ious = diou_loss(cls_bbox, s_cls_bbox)
                    diou_losses = diou_losses.cpu().numpy()
                    bin_idx = int(np.digitize(diou_losses, cls_dist['bin_edges']) - 1)
                    bin_idx = min(bin_idx, len(cls_dist['bin_edges']) - 2)
                    local_dist[s_cls] = torch.tensor((cls_dist['bin_edges'][bin_idx], cls_dist['bin_edges'][bin_idx + 1]))
            
        mix_axis, ori_axis, global_ = seg_sliding_windows(
            param=param,
            source_cls=cls.item(), 
            cls_mask=cls_mask,
            gt_mask=target_gt,
            local_dist=local_dist,
            cls_dist_mat=cls_dist_mat,
            cls_relation=cls_dist['relation'],
        )

        masked_img = source_img * cls_mask
        masked_gt = source_gt * cls_mask
        masked_weight = source_weight * cls_mask

        if mix_axis is not None:
            ori_x1, ori_y1, ori_x2, ori_y2 = ori_axis
            mix_x1, mix_y1, mix_x2, mix_y2 = mix_axis

            ori_target_img = target_img.clone()

            mix_img = masked_img[:, ori_y1:ori_y2, ori_x1:ori_x2]
            target_img[:, mix_y1:mix_y2, mix_x1:mix_x2].copy_(
                torch.where(mix_img != 0, mix_img, target_img[:, mix_y1:mix_y2, mix_x1:mix_x2])
            )

            ori_target_gt = target_gt.clone()
            
            mix_gt = masked_gt[ori_y1:ori_y2, ori_x1:ori_x2]
            target_gt[mix_y1:mix_y2, mix_x1:mix_x2].copy_(
                torch.where(mix_gt != 0, mix_gt, target_gt[mix_y1:mix_y2, mix_x1:mix_x2])
            )

            mix_weight = masked_weight[ori_y1:ori_y2, ori_x1:ori_x2]
            target_weight[mix_y1:mix_y2, mix_x1:mix_x2].copy_(
                torch.where(mix_weight != 0, mix_weight, target_weight[mix_y1:mix_y2, mix_x1:mix_x2])
            )

            if global_:
                vis_mix_cls = masked_gt.clone()
                vis_mix_cls[vis_mix_cls == 0] = 255
                vis_mixing_cls(
                    torch.clamp(denorm(source_img, param['mean'], param['std']), 0, 1)[0], 
                    torch.clamp(denorm(masked_img, param['mean'], param['std']), 0, 1)[0], 
                    torch.clamp(denorm(ori_target_img, param['mean'], param['std']), 0, 1)[0], 
                    torch.clamp(denorm(target_img, param['mean'], param['std']), 0, 1)[0], 
                    source_gt, vis_mix_cls, ori_target_gt, target_gt
                )
                print('global_mix')
        else:

            # using ori_axis to mix for invalid cls         
            target_img.copy_(
                torch.where(masked_img != 0, masked_img, target_img)
            )

            target_gt.copy_(
                torch.where(masked_gt != 0, masked_gt, target_gt)
            )

            target_weight.copy_(
                torch.where(masked_weight != 0, masked_weight, target_weight)
            )
        
    return target_img.unsqueeze(0), target_gt.unsqueeze(0).unsqueeze(0), target_weight.unsqueeze(0).unsqueeze(0)


def strong_transform(param, data=None, target=None, weight=None, cls_dist=None):
    """
        Args:
            param: dict, parameters for strong transform
            data: tensor, images, shape (2, C, H, W), where 2 is source and target
            target: tensor, groundtruths, shape (2, H, W), where 2 is source and target
            weight: tensor, weights for pseudo labels, shape (2, H, W), where 2 is source and target

        Returns:
            mixed_data: tensor, images, shape (2, C, H, W), where 2 is source and target
            mixed_target: tensor, groundtruths, shape (2, H, W), where 2 is source and target
            mixed_weight: tensor, weights for pseudo labels, shape (2, H, W), where 2 is source and target
    """
    assert ((data is not None) or (target is not None))

    if cls_dist is not None:
        mixed_data, mixed_target, mixed_weight = cls_dist_mix(
            param=param,
            data=data, target=target, weight=weight, cls_dist=cls_dist,
        )
    else:
        mixed_data, mixed_target = one_mix(mask=param['mix'], data=data, target=target)
        _, mixed_weight = one_mix(mask=param['mix'], data=None, target=weight)

    # mixed_data, mixed_target = color_jitter(
    #     color_jitter=param['color_jitter'],
    #     s=param['color_jitter_s'],
    #     p=param['color_jitter_p'],
    #     mean=param['mean'],
    #     std=param['std'],
    #     data=mixed_data,
    #     target=mixed_target)
    # mixed_data, mixed_target = gaussian_blur(blur=param['blur'], data=mixed_data, target=mixed_target)
    
    return mixed_data, mixed_target, mixed_weight


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


def get_class_masks(labels, class_choice=None):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        if class_choice is None:
            class_choice = np.random.choice(
                nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        else:
            class_choice = [i for i in class_choice if i in classes]
            class_choice = [torch.where(classes == i)[0].item() for i in class_choice]

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
        # 'configs/generated/local-exp86/240711_1747_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_902ca.py'
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
        
        mix_masks = get_class_masks(data['gt_semantic_seg'].data, class_choice=[17])
        
        strong_parameters['mix'] = mix_masks[0]
        strong_parameters['topk'] = 2

        img = torch.stack((data['img'].data, data['target_img'].data))
        target = torch.stack((data['gt_semantic_seg'].data[0], data['target_gt'].data[0]))
        
        strong_parameters['dist_mode'] = ['local', 'global']
        local_img, local_mixed_target, mixed_weight = strong_transform(
            strong_parameters,
            data=img,
            target=target,
            weight=torch.ones_like(target),
            cls_dist=data['cls_dist'],
        )


        # img = torch.stack((data['img'].data, data['target_img'].data))
        # target = torch.stack((data['gt_semantic_seg'].data[0], data['target_gt'].data[0]))

        # strong_parameters['dist_mode'] = ['global']
        # global_img, global_mixed_target, mixed_weight = strong_transform(
        #     strong_parameters,
        #     data=img,
        #     target=target,
        #     weight=torch.ones_like(target),
        #     cls_dist=data['cls_dist'],
        # )

        # img = torch.stack((data['img'].data, data['target_img'].data))
        # target = torch.stack((data['gt_semantic_seg'].data[0], data['target_gt'].data[0]))

        # classmix_img, mixed_target, mixed_weight = strong_transform(
        #     strong_parameters,
        #     data=img,
        #     target=target,
        #     weight=torch.ones_like(target),
        # )

        # vis_mixing_cls(
        #     torch.clamp(denorm(data['img'].data, means, stds), 0, 1)[0], 
        #     torch.clamp(denorm(data['target_img'].data, means, stds), 0, 1)[0], 
        #     torch.clamp(denorm(local_img, means, stds), 0, 1)[0], 
        #     torch.clamp(denorm(global_img, means, stds), 0, 1)[0],
        #     data['gt_semantic_seg'].data[0],
        #     data['target_gt'].data[0],
        #     local_mixed_target[0][0],
        #     global_mixed_target[0][0],
        # )
        # print('here')