import pickle

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm


def visualize(prob, cls_name, num_cls):
    # 确定 a 和 b 的范围
    columns = 5  # 每排5个图表

    # 对其中一个 cls_prob 进行可视化
    for a in tqdm(range(num_cls)):
        rows = (num_cls + columns - 1) // columns  # 计算需要的行数
        fig, axs = plt.subplots(rows, columns, figsize=(columns * 4, rows * 5))
        fig.suptitle(f'Probability Distributions for cls: {cls_name[a]}', fontsize=20)
        
        x_values = np.arange(0, 2, 0.1)
        x_color = np.linspace(1, 0, 20)
        cm = plt.cm.get_cmap('RdYlBu_r')
        
        width = 0.1  # 设置条形的宽度为0.1
        
        for b in range(num_cls):
            row = b // columns
            col = b % columns
            axs[row, col].bar(x_values, prob[a, b], width=width, color=cm(x_color), edgecolor='black')
            axs[row, col].set_title(f'({cls_name[a]}, {cls_name[b]})', fontsize=14)
            axs[row, col].set_ylim(0, max(prob[a, b]) * 1.5 + 1e-6)  # 设置 y 轴的范围
            axs[row, col].set_xlim(0, 2)  # 设置 x 轴的范围为 0 到 2
            axs[row, col].set_xticks(np.arange(0, 2.1, 0.5))  # 设置 x 轴刻度为 0 到 2，间隔为 0.5

        # 如果 b_range 不能被 columns 整除，需要隐藏多余的子图
        for b in range(num_cls, rows * columns):
            row = b // columns
            col = b % columns
            fig.delaxes(axs[row, col])

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 调整标题位置
        os.makedirs('tools/classes_distance/fig', exist_ok=True)
        plt.savefig(f'tools/classes_distance/fig/cls{a}_{cls_name[a]}_prob.jpg')
        plt.close(fig)  # 关闭图像以释放内存


def prob_dist(cls_a, distribution, num_cls):
    distance = []
    for cls_b in range(num_cls):
        d = distribution[cls_a, cls_b]
        indices = np.arange(len(d))
        distance.append(np.sum(indices * d))

    return distance


if __name__ == '__main__':
    with open('data/cityscapes/cls_prob_distribution_diou.pkl', 'rb') as file:
        cls_dist = pickle.load(file)

    cls_name = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle',
    ]
    
    prob = cls_dist['prob']
    num_cls = 19

    for cls_a in range(num_cls):
        prob_dist(cls_a, prob, num_cls)

    visualize(prob, cls_name, num_cls)

