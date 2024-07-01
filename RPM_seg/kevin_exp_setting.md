## Environment Setup
```shell
conda create -n rpm python=3.8.5 pip=22.3.1
conda activate rpm
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```

## Run Best Result From Kevin
```shell
python run_experiments.py --config configs/rpm/gtaHR2csHR_mic_hrda.py
```

## Best Config From Kevin's Exp

```python
# masking
mask_type = 'proto_prob', 
mask_ratio = 0.6

# rare class mixing
rcs_class_temp = 0.5
rare_class_mix = True

# 調整𝒕_𝒑𝒓𝒐𝒃: masking_transforms.py/ def generate_proto_prob_mask(): 
confidence = torch.softmax(confidence/ 0.1, dim=1)

# Curriculum Learning
usedCL = False
```

## Config Setting
### configs/rpm/gtaHR2csHR_mic_hrda.py
```python
# masking
mask_type = 'original', 'original_pixelwise', 'proto', 'proto_prob', 
mask_ratio = 0.5 (設定masking比率)

# rare class mixing
rcs_class_temp = 0.5
rare_class_mix = True

# 調整𝒕_𝒑𝒓𝒐𝒃: masking_transforms.py/ def generate_proto_prob_mask(): 
confidence = torch.softmax(confidence/ 0.1, dim=1)

# Curriculum Learning
usedCL = True
r_0 = 0.6,
r_final= 0.7,
total_iteration = 40000
```

## Dataset
dataset: '/mnt/Nami/rayeh/data'

預設dataset: GTA->Cityscapes

換成Synthia->Cityscapes:
uda_synthiaHR_to_cityscapesHR_1024x1024.py 裡面的 data_root要正確