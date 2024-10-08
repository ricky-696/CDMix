# DCMix for Domain-Adaptive Semantic Segmentation

## Environment Setup

```shell
conda create -n mic python=3.8.5 pip=22.3.1
conda activate mic
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```

Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

## Dataset Setup

> [!TIP]
> For MVL: The complete dataset (including all data preprocessing) is located at: `/mnt/Nami/dataset/MIC_dataset`

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia (Optional):** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.

**ACDC (Optional):** Please, download rgb_anon_trainvaltest.zip and
gt_trainval.zip from [here](https://acdc.vision.ee.ethz.ch/download) and
extract them to `data/acdc`. Further, please restructure the folders from
`condition/split/sequence/` to `split/` using the following commands:

```shell
rsync -a data/acdc/rgb_anon/*/train/*/* data/acdc/rgb_anon/train/
rsync -a data/acdc/rgb_anon/*/val/*/* data/acdc/rgb_anon/val/
rsync -a data/acdc/gt/*/train/*/*_labelTrainIds.png data/acdc/gt/train/
rsync -a data/acdc/gt/*/val/*/*_labelTrainIds.png data/acdc/gt/val/
```

**Dark Zurich (Optional):** Please, download the Dark_Zurich_train_anon.zip
and Dark_Zurich_val_anon.zip from
[here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) and extract it
to `data/dark_zurich`.

The final folder structure should look like this:

```none
DAFormer
├── ...
├── data
│   ├── acdc (optional)
│   │   ├── gt
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── dark_zurich (optional)
│   │   ├── gt
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia (optional)
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## Calaulating Class Distance
Before training, we must use the following command to calculate the Class Distance from the labeled source domain.

```shell
python seg/tools/classes_distance/cnt_cls_dist.py --config configs/_base_/datasets/uda_gta_to_cityscapes_512x512.py --dataset source target
```

The dataset's config is saved in `configs/_base_/datasets`, you can convert only the source domain or target domain using the `--dataset` argument; the default is source and target.

After executing this Python file, you will get the `cls_dists_diou.pkl` and `cls_prob_distribution_diou.pkl` in the dataset's folder.

## Training
For the experiments from rpm & DCMix, we used the config file from `configs/rpm` and the following command:

```shell
python run_experiments.py --config <config path>
```

Suppose you need to change the dataset path. The path must be modified in the `data_root` variables for both source and target in the config file from `RPM_seg/configs/_base_/datasets`.

## Evaluation

A trained model can be evaluated using:

```shell
sh test.sh work_dirs/run_name/
```

The predictions are saved for inspection to
`work_dirs/run_name/preds`
and the mIoU of the model is printed to the console.

When training a model on Synthia→Cityscapes, please note that the
evaluation script calculates the mIoU for all 19 Cityscapes classes. However,
Synthia contains only labels for 16 of these classes. Therefore, it is a common
practice in UDA to report the mIoU for Synthia→Cityscapes only on these 16
classes. As the Iou for the 3 missing classes is 0, you can do the conversion
`mIoU16 = mIoU19 * 19 / 16`.

The results for Cityscapes→ACDC and Cityscapes→DarkZurich are reported on
the test split of the target dataset. To generate the predictions for the test
set, please run:

```shell
python -m tools.test path/to/config_file path/to/checkpoint_file --test-set --format-only --eval-option imgfile_prefix=labelTrainIds to_label_id=False
```

The predictions can be submitted to the public evaluation server of the
respective dataset to obtain the test score.

## MIC's Checkpoints

Below, we provide checkpoints of MIC(HRDA) for the different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the median validation performance here.

* [MIC(HRDA) for GTA→Cityscapes](https://drive.google.com/file/d/1p_Ytxmj8EckYsq6SdZNZJNC3sgxVRn2d/view?usp=sharing)
* [MIC(HRDA) for Synthia→Cityscapes](https://drive.google.com/file/d/1-Ed0Z2APrhIdsuQTOWXNlZwJJ9Yr2-Vu/view?usp=sharing)
* [MIC(HRDA) for Cityscapes→ACDC](https://drive.google.com/file/d/10RNOAyUY5nYKzIIbNTie458r9etzfvtc/view?usp=share_link)
* [MIC(HRDA) for Cityscapes→DarkZurich](https://drive.google.com/file/d/1HXIwLULUsspBG4U1UAd7OQnDq1G33aTA/view?usp=sharing)

The checkpoints come with the training logs. Please note that:

* The logs provide the mIoU for 19 classes. For Synthia→Cityscapes, it is
  necessary to convert the mIoU to the 16 valid classes. Please, read the
  section above for converting the mIoU.
* The logs provide the mIoU on the validation set. For Cityscapes→ACDC and
  Cityscapes→DarkZurich the results reported in the paper are calculated on the
  test split. For DarkZurich, the performance significantly differs between
  validation and test split. Please, read the section above on how to obtain
  the test mIoU.

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for DCMix & MIC are:

* [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py):
  Implementation of the DAFormer/HRDA/MIC self-training pipeline
* [mmseg/datasets/uda_dataset.py](mmseg/datasets/uda_dataset.py): Implementation of the DCMix's class relation & dataloader
* [mmseg/models/utils/dacs_transforms.py](mmseg/models/utils/dacs_transforms.py): Implementation of the DCMix's mixing process
* [configs/mic/gtaHR2csHR_mic_hrda.py](configs/mic/gtaHR2csHR_mic_hrda.py):
  Annotated config file for MIC(HRDA) on GTA→Cityscapes.
* [experiments.py](experiments.py): Definition of the experiment configurations in the paper.
* [mmseg/models/uda/masking_consistency_module.py](mmseg/models/uda/masking_consistency_module.py): Implementation of MIC.
* [mmseg/models/utils/masking_transforms.py](mmseg/models/utils/masking_transforms.py): Implementation of the image patch masking.


## Acknowledgements

DCMix is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [MIC](https://github.com/lhoyer/MIC)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
